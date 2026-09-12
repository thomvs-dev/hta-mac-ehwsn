"""Passive packet birth ledger; never changes service or random draws."""
import numpy as np


class ArrivalRecorder:
    def __init__(self,rng):self.rng=rng;self.arrivals=[]
    def __getattr__(self,name):return getattr(self.rng,name)
    def poisson(self,*args,**kwargs):
        value=self.rng.poisson(*args,**kwargs)
        self.arrivals.append(int(value));return value


class PacketCohort:
    def __init__(self,base,first_birth,last_birth):
        self.first=first_birth;self.last=last_birth
        self.births=[[base.round-age for age in ages] for ages in base.packet_ages]
        self.counts={key:np.zeros(base.n_nodes,dtype=np.int64) for key in ['generated','delivered','stale','death','overflow']}
        self.delay_hist={};self.recorder=ArrivalRecorder(base.np_random);base.np_random=self.recorder
        for i,births in enumerate(self.births):self.counts['generated'][i]=sum(self.contains(b) for b in births)

    def contains(self,birth):return self.first<=birth<=self.last

    def observe(self,base,served):
        now=int(base.round);arrivals=iter(self.recorder.arrivals)
        assert len(self.recorder.arrivals)==int(base.alive.sum())
        for node,births in enumerate(self.births):
            n=int(served[node]);assert n<=len(births)
            for birth in births[:n]:
                if self.contains(birth):
                    self.counts['delivered'][node]+=1
                    delay=now-birth;self.delay_hist[delay]=self.delay_hist.get(delay,0)+1
            left=births[n:]
            if not base.alive[node]:
                self.counts['death'][node]+=sum(self.contains(b) for b in left);left=[]
            else:
                expired=[b for b in left if now-b>base.cfg.packet_ttl_rounds]
                self.counts['stale'][node]+=sum(self.contains(b) for b in expired)
                left=[b for b in left if now-b<=base.cfg.packet_ttl_rounds]
                count=next(arrivals)
                if self.contains(now):self.counts['generated'][node]+=count
                left.extend([now]*count)
                overflow=max(0,len(left)-base.cfg.queue_max_packets)
                self.counts['overflow'][node]+=sum(self.contains(b) for b in left[:overflow])
                left=left[overflow:]
            self.births[node]=left
            assert [now-b for b in left]==base.packet_ages[node]
        self.recorder.arrivals.clear()
        pending=np.array([sum(self.contains(b) for b in births) for births in self.births])
        assert np.array_equal(self.counts['generated'],pending+sum(self.counts[k] for k in ['delivered','stale','death','overflow']))

    def result(self):
        totals={key:int(value.sum()) for key,value in self.counts.items()}
        totals['pending']=sum(sum(self.contains(b) for b in births) for births in self.births)
        totals['delivery_probability']=totals['delivered']/max(1,totals['generated'])
        totals['stale_probability']=totals['stale']/max(1,totals['generated'])
        totals['delivered_delay_histogram_rounds']=self.delay_hist
        totals['per_node']={k:v.tolist() for k,v in self.counts.items()}
        return totals
