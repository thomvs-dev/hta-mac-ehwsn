"""Versioned scheduled-control simulation, separate from frozen V1 and R0.

Control and ACK channels are reliable conditional on local energy funding.
Data/forward erasures consume their full reserved radio cost; ACK listening
and negative ACKs are charged even after failure. No ACK-loss claim is made.
"""
import copy
import numpy as np
from experiments.evaluate_step4_publication_evidence import ScenarioRoleSeparatedScheduledMACEnv
from envs.reviewer_energy_feasibility_fast import CostTable


class KeyedTraffic:
    def begin(self,seed,frame,alive,rate):
        self.values=iter(np.random.default_rng([seed,frame,101]).poisson(rate,len(alive))[alive])
    def poisson(self,rate):return int(next(self.values))


class PaidTable(CostTable):
    def __init__(self,base,bits):
        super().__init__(base)
        self.ack_member=np.zeros_like(self.member);self.ack_head=np.zeros_like(self.head)
        for ci,ch in enumerate(base.cluster_heads):
            members=np.flatnonzero((base.cluster_of==ci)&base.alive&(np.arange(base.n_nodes)!=ch))
            distance=max((float(np.linalg.norm(base.positions[n]-base.positions[ch])) for n in members),default=0.)
            for total in range(1,self.b+1):
                self.ack_head[ci,total]=total*base.radio.tx(bits,distance)
                for n in members:
                    for k in range(1,total+1):self.ack_member[n,total,k]=k*base.radio.rx(bits)
        self.member+=self.ack_member;self.head+=self.ack_head


class PaidControlMAC(ScenarioRoleSeparatedScheduledMACEnv):
    def __init__(self,*args,data_error=0.,control_bits=100,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<=data_error<=1:raise ValueError('invalid erasure probability')
        if type(control_bits) is not int or control_bits<=0:raise ValueError('invalid control bits')
        self.data_error=data_error;self.control_bits=control_bits

    def reset(self,*,seed,frozen_snapshot):
        result=super().reset(seed=seed,frozen_snapshot=frozen_snapshot)
        self.seed=int(seed);self.traffic=KeyedTraffic();self.np_random=self.traffic
        self.last_harvest=np.zeros(self.n_nodes);self.control_ready=False
        return result

    def _exchange(self,sender,receiver,cost):
        d=float(np.linalg.norm(self.positions[sender]-self.positions[receiver]))
        tx=self.radio.tx(self.control_bits,d);rx=self.radio.rx(self.control_bits)
        # Independently scheduled receiver listening is paid even if sender
        # cannot transmit. Only local current energy authorizes each operation.
        sent=self.alive[sender] and self.energy[sender]>=tx
        heard=self.alive[receiver] and self.energy[receiver]>=rx
        if sent:self.energy[sender]-=tx;cost[sender]+=tx
        if heard:self.energy[receiver]-=rx;cost[receiver]+=rx
        return bool(sent and heard)

    def begin_control(self):
        if self.control_ready:raise RuntimeError('control phase already reserved')
        self.frame_energy_before=self.energy.copy();self.frame_alive_before=self.alive.copy()
        self.report_cost=np.zeros(self.n_nodes);self.grant_cost=np.zeros(self.n_nodes);accepted=np.zeros(self.n_nodes,bool)
        self.report_attempts=0;self.report_success=0
        for ci,ch in enumerate(self.cluster_heads):
            members=sorted(np.flatnonzero((self.cluster_of==ci)&(np.arange(self.n_nodes)!=ch)),key=lambda n:tuple(self.positions[n]))
            for node in members:
                self.report_attempts+=1
                reported=self._exchange(node,ch,self.report_cost)
                self.report_success+=int(reported)
                # A known per-member grant slot: reserve its fixed cost now;
                # contents are filled after allocation, without further energy.
                granted=self._exchange(ch,node,self.grant_cost)
                accepted[node]=reported and granted
        accepted[self.cluster_heads]=self.alive[self.cluster_heads]
        self.control_cost=self.report_cost+self.grant_cost
        view=copy.copy(self)
        view.alive=self.alive&accepted;view.queue=self.queue.copy();view.queue[~accepted]=0
        view.energy=self.energy.copy();view.energy[~accepted]=0
        view.packet_ages=[list(a) if accepted[n] else [] for n,a in enumerate(self.packet_ages)]
        # Expose only the explicit scheduler schema, never simulator RNG/state.
        from types import SimpleNamespace
        schema={k:getattr(view,k) for k in ['n_nodes','positions','cluster_heads','cluster_of','alive','queue','energy','packet_ages','cfg','radio','idle_energy_enabled']}
        self.accepted=accepted;self.control_ready=True
        self.view=SimpleNamespace(**schema)
        self.paid_table=PaidTable(self.view,self.control_bits)
        return self.view,self.paid_table

    def _sample_harvest(self):
        rng=np.random.default_rng([self.seed,self.round,102]);result=np.zeros(self.n_nodes)
        for hmm,states,scale in [(self.solar,self.solar_states,self.cfg.solar_scale),(self.thermal,self.thermal_states,self.cfg.thermal_scale)]:
            result+=np.maximum(0.,rng.normal(hmm.mean[states],np.sqrt(hmm.variance[states]))*scale)
            cdf=np.cumsum(hmm.transition[states],axis=1)
            states[:]=np.sum(rng.random(self.n_nodes)[:,None]>cdf,axis=1)
        result[~self.alive]=0.
        return result*self.harvest_multiplier

    def _update_queues(self,delivered):
        self.traffic.begin(self.seed,self.round,self.alive,self.arrival_rate)
        super()._update_queues(delivered)

    def step(self,action):
        if not self.control_ready:raise RuntimeError('begin_control required')
        a=np.asarray(action)
        if not np.issubdtype(a.dtype,np.integer):raise ValueError('integer action required')
        self._validate_action(a)
        if np.any(a>self.view.queue) or np.any(a[~self.accepted]):raise ValueError('unreported or unavailable queue used')
        data_ack=self.paid_table.costs(a)[0]
        if np.any(data_ack>self.energy+1e-15):raise ValueError('unfunded service')
        delivered=np.zeros(self.n_nodes,dtype=np.int64)
        # Fixed-size exogenous vectors: unused draws never shift other nodes.
        for ci,ch in enumerate(self.cluster_heads):
            forward=np.random.default_rng([self.seed,self.round,int(ch),104]).random()>=self.data_error
            for node in np.flatnonzero((self.cluster_of==ci)&(a>0)):
                draws=np.random.default_rng([self.seed,self.round,int(node),103]).random(self.cfg.n_max)
                # Each slot retries the oldest unreceived packet; success moves
                # to the next FIFO packet. Failed aggregate forwarding retains
                # the whole source prefix for a later frame; reliable ACKs.
                delivered[node]=int((draws[:a[node]]>=self.data_error).sum()) if forward else 0
        role={k:np.zeros(self.n_nodes) for k in ['member_tx','ch_rx','ch_aggregate','ch_forward','data_idle','ack']}
        for ci,ch in enumerate(self.cluster_heads):
            nodes=np.flatnonzero((self.cluster_of==ci)&(a>0));total=int(a[nodes].sum())
            if not total:continue
            for node in nodes:
                role['member_tx'][node]=self.radio.tx(int(a[node])*self.cfg.packet_bits,float(np.linalg.norm(self.positions[node]-self.positions[ch])))
                if self.idle_energy_enabled:role['data_idle'][node]=(total-int(a[node]))*self.cfg.e_elec_j_per_bit*self.cfg.idle_slot_bit_times
            role['ch_rx'][ch]=self.radio.rx(total*self.cfg.packet_bits)
            role['ch_aggregate'][ch]=self.radio.aggregate(total*self.cfg.packet_bits)
            role['ch_forward'][ch]=self.radio.tx(self.cfg.packet_bits,float(np.linalg.norm(self.positions[ch]-np.asarray(self.cfg.bs_position_m))))
        original=sum(role.values());role['ack']=data_ack-original
        if np.any(role['ack'] < -1e-14):raise RuntimeError('negative ACK reconstruction')
        np.testing.assert_allclose(sum(role.values()),data_ack,rtol=0,atol=1e-14)
        self.total_idle_energy+=float(role['data_idle'].sum())
        harvested=self._sample_harvest();consumed=self.control_cost+data_ack
        self.energy=np.minimum(self.cfg.initial_energy_j,np.maximum(0.,self.energy-data_ack+harvested))
        self.alive=self.frame_alive_before&(self.energy>0)
        self._update_queues(delivered);self.previous_slots=a.copy();self.round+=1
        self.total_packets+=int(delivered.sum())
        if self.t_fnd is None and not self.alive.all():self.t_fnd=self.round
        self.last_harvest=harvested.copy();self.control_ready=False
        terminated=not self.alive.any();truncated=self.round>=self.cfg.max_rounds
        if not terminated and not truncated:truncated=not self._install_schedule_round(self.round)
        info=self._info()
        info.update(delivered_packets_per_node=delivered,attempted_packets_per_node=a.copy(),energy_trace=dict(consumed=consumed,control=self.control_cost.copy(),report=self.report_cost.copy(),grant=self.grant_cost.copy(),data_ack=data_ack,role_energy=role,harvested=harvested,energy_before=self.frame_energy_before.copy(),energy_after=self.energy.copy()),report_attempts=self.report_attempts,report_success=self.report_success,accepted_members=int(self.accepted.sum()-self.frame_alive_before[self.view.cluster_heads].sum()))
        return self._state(),0.,bool(terminated),bool(truncated),info
