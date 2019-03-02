from particle_filter import *

class part_filt_variable_window(part_filt):

    def __init__(self, num, temp, w, h, sig_d, sig_mse, beta, alpha = 0):
        part_filt.__init__(self,num, temp, w, h, sig_d, sig_mse, alpha = alpha)
        self.beta = beta
        self.first_time = True
        self.ht, self.wdt= temp.shape[:2]
        self.a_pha = self.wdt*1.0/self.ht

    def sample(self, frame):
        part_filt.sample(self,frame)

        if not self.first_time:
            dm = int(self.beta * self.wdt)
            dn = int(self.beta * self.ht)
            u = int(np.mean(np.array([self.xt_1[i].u for i in xrange(10)])))
            v = int(np.mean(np.array([self.xt_1[i].v for i in xrange(10)])))
            n = self.ht
            m = self.wdt

            grad_m = (-self.MSE(frame[v:v+n,u-dm:u+m-dm],frame[v:v+n,u:u+m])+self.MSE(frame[v:v+n,u+dm:u+m+dm],frame[v:v+n,u:u+m]))/2/dm 
            grad_n = (-self.MSE(frame[v-dn:v+n-dn,u:u+m],frame[v:v+n,u:u+m])+self.MSE(frame[v+dn:v+dn+n,u:u+m],frame[v:v+n,u:u+m]))/2/dn

            if self.a_pha*grad_m + grad_n > 0:
                self.wdt -= dm
                self.ht -= int(self.a_pha*dm)
                self.temp = cv2.resize(self.temp, (self.wdt,self.ht), interpolation = cv2.INTER_CUBIC)
            elif self.a_pha*grad_m + grad_n <= -1:
                self.wdt += dm
                self.ht += int(self.a_pha*dm)
                self.temp = cv2.resize(self.temp, (self.wdt,self.ht), interpolation = cv2.INTER_CUBIC)

        print 'width = ', self.wdt,' height = ',self.ht
        self.first_time = False
