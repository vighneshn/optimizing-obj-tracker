# optimizing-obj-tracker

run check.py
edit particle_filter.py to optimize code, lines 77 to 100.
I have already compressed it by 50 lines, to no performance bonus, still takes ~0.06s per frame, need to better that by better vectorization of the call to self.pzt()

2x speed up in the second proper commit, where pzt was parallelized, would've hoped for even greater speed up, but shall settle for what i have. probably, if i reduce 3 calls to pzt into 1, that should be perfect.

optimised code to some extent. still takes 0.05s for a template of size 123x102, used to take 0.1s, hence still slow, have to look for other gains. Not the 10-20x gain HSR was talking about.
