import numpy as np
import cv2
import time
import scipy.linalg.blas as slb
from itertools import *
from mergeSort import *


#Particle Filter class
class part_filt:

	#COnstructor
	def __init__(self, num, temp, w, h, sig_d, sig_mse, init_center, sigma_wm = 1, ff = 0.9, n_0 = 6, k = 10, alpha = 0.7):
		self.num = num
		self.n, self.m = temp.shape[:2]
		self.frames_passed = 0
		self.forget_factor = ff
		self.mu_data = temp.reshape((temp.size, 1))

		#number of eigenvectors 
	        self.k = k
                self.predicted = []

		self.sub_s = np.zeros((self.n*self.m, 0), order='C') 
		self.sigma_svd = np.zeros(0)

		self.xt = []
		self.xt_1 = []

		self.n_0 = n_0
		self.prev_us = np.zeros((0,1))
		self.prev_vs = np.zeros((0,1))
		self.t_poly_weights = np.zeros((4,2))
		self.t_matrix = np.zeros((0,4))

		self.sig_mse = sig_mse
		self.sig_d = sig_d
		self.alpha = alpha
		self.sigma_wm = sigma_wm

		self.weightmask = np.ones(self.mu_data.shape)
		for j in xrange(self.num):
			x = particle(init_center[0], init_center[1], 1.0/self.num)
			self.xt_1.append(x)
	
	#Sample only first 'n' particles with highest weight
	#This solves the issue. LOL :P
	def sample(self, frame):
                start = time.clock()

	        self.num = len(self.xt_1)
		total_p = 0 
		i = 0
		eta = 0.0

		#Store the mean of the ten best particles
		nmlz = np.sum([self.xt_1[k].wt for k in range(10)])
		self.mean_best_ten = (np.sum([self.xt_1[k].u*self.xt_1[k].wt for k in range(10)])/nmlz, 
							  np.sum([self.xt_1[k].v*self.xt_1[k].wt for k in range(10)])/nmlz)

		self.regress()
		u_t_plus_1 = self.get_new_u()
		v_t_plus_1 = self.get_new_v()
                self.predicted.append((int(u_t_plus_1),int(v_t_plus_1)))
		self.print_vel()
		
		n_vel = int(self.alpha*self.num) #n particles with the velocity

                ####TRYING TO COMPRESS CODE, now need to vectorize calling pzt, instead of inside the forloop.
                p = [int(round(self.xt_1[i].wt*n_vel,0)) for i in range(n_vel)]
                delu = lambda x:np.random.normal(u_t_plus_1 - self.mean_best_ten[0], self.sig_d, p[x]) if p[x]>0 else [0]
                delv = lambda x:np.random.normal(v_t_plus_1 - self.mean_best_ten[1], self.sig_d, p[x]) if p[x]>0 else [0]
                delu2 = lambda x:np.random.normal(u_t_plus_1 - self.mean_best_ten[0], self.sig_d, n_vel - sum(p) +p[x]) if p[x]>0 else [0]
                delv2 = lambda x:np.random.normal(v_t_plus_1 - self.mean_best_ten[1], self.sig_d, n_vel - sum(p) +p[x]) if p[x]>0 else [0]

                u_v = [[[self.xt_1[i].u+delu(j)[0], self.xt_1[i].v+delv(j)[0]] for j in range(p[i])] if sum(p[:i])<=n_vel else [[self.xt_1[i].u+delu2(j)[0], self.xt_1[i].v+delv2(j)[0]] for j in range(n_vel - sum(p) + p[i])] for i in range(n_vel)]
                u_v = np.asarray(list(chain.from_iterable(u_v)))

                if sum(p) < n_vel:
                    delu = np.random.normal(u_t_plus_1 - self.mean_best_ten[0], self.sig_d, n_vel - sum(p))
                    delv = np.random.normal(v_t_plus_1 - self.mean_best_ten[1], self.sig_d, n_vel - sum(p))
                    u_v = np.concatenate((u_v,np.asarray([[self.xt_1[i].u+delu[i], self.xt_1[i].v+delv[i]] for i in range(n_vel - sum(p))])), axis=0)

                ##Without velociy
                _del = np.random.normal(0, self.sig_d, self.num - n_vel)
                u_v = np.concatenate((u_v,np.asarray([[self.xt_1[i].u+_del[i], self.xt_1[i].v+_del[i]] for i in range(self.num - n_vel)])), axis = 0)

                weights = self.pzt(frame, u_v[:,0], u_v[:,1])
                self.xt = [particle(u_v[i][0],u_v[i][1], weights[i]) for i in range(self.num)]

                wt = sum([self.xt[i].wt for i in range(len(self.xt))])
                for i in range(len(self.xt)):
                    self.xt[i].wt /= wt
		
		#Merge sort, to sort particles by weight
		self.sort_by_weight()
		self.xt_1 = self.xt
		self.xt = []
				
		self.update_temp(frame)
		#self.disp_eig()
		self.frames_passed = self.frames_passed + 1
		print 'sample function time ',time.clock() - start 
	
	#Calculate P(Zt|Xt)
	def pzt(self, frame, u, v):
		h,w = frame.shape[:2]
		start = time.clock()
		#Boundary Condtitions... :P
                # Need to make checks for the boundary in this method
		#if(u<=w-self.m/2 and u >= self.m/2 and v >= self.n/2 and v <= h-self.n/2):
                img2 = np.asarray([frame[int(round(v[i] - self.n/2.0,0)): int(round(v[i] + self.n/2.0,0)), int(round(u[i] - self.m/2.0,0)): int(round(u[i]+self.m/2.0,0))] for i in range(u.size)])
                img2 = img2.reshape((u.size,self.n*self.m))
                err = self.MSE(img2.T)
                weight = np.exp(-err/(2*self.sig_mse**2))
		print 'pzt time ', time.clock() - start
                return weight

        #MSE with robust error norm
        def MSE(self,img2):
                start = time.clock()
                z = img2 - self.mu_data
                #if self.frames_passed >= 1:
                #    p = slb.dgemm(1.0, a=self.sub_s, b=slb.dgemm(1.0 ,a=self.sub_s.T, b=z))
                #else:
                p = np.dot(self.sub_s, np.dot(self.sub_s.T,z))
                l = (z-p)**2
                err = np.sum((l.astype(float)/(l+(38**2)*3)), axis = 0)
                return err

	def sort_by_weight(self):
		mergeSort(self.xt, 0, self.num-1)

		
	def update_temp(self, frame):
                #
		nmlz = np.sum([self.xt_1[i].wt for i in range(10)])
		u = np.sum([self.xt_1[i].u*self.xt_1[i].wt for i in range(10)])/nmlz 
		v = np.sum([self.xt_1[i].v*self.xt_1[i].wt for i in range(10)])/nmlz
                #
                img2 =frame[int(round(v - self.n/2.0,0)): int(round(v + self.n/2.0,0)), int(round(u - self.m/2.0,0)): int(round(u+self.m/2.0,0))]
		img2 = img2.reshape((img2.size,1))

		B = img2
		factor = (self.frames_passed*1.0/(self.frames_passed + 1 ))**0.5

		B_hat = np.append( np.zeros((img2.size,1)), (img2 - self.mu_data) * factor , axis = 1)
		self.mu_data = (self.mu_data*(self.frames_passed)*self.forget_factor + img2)*1./((self.frames_passed)*self.forget_factor+1)

		U_sigma = self.forget_factor*np.dot(self.sub_s, np.diag(self.sigma_svd)) #Matrix multiplication of U and Sigma
		QR_mat = np.append(U_sigma, B_hat, axis = 1) #This is the matrix whose QR factors we want
		U_B_tild, R = np.linalg.qr(QR_mat)
		
		U_tild, sig_tild, vh_tild = np.linalg.svd(R)
		U_new = np.dot(U_B_tild, U_tild )
		if(sig_tild.size > self.k):
			self.sigma_svd = sig_tild[ 0:self.k ]
			self.sub_s = U_new[:, 0:self.k ]
		else:
			j = 0 #iterator
			while j < self.sub_s.shape[1]:
				self.sub_s[:,j] = U_new[:,j]
				self.sigma_svd[j] = sig_tild[j]
				j = j+1
			self.sub_s = np.append(self.sub_s, U_new[:,j].reshape((self.sub_s.shape[0], 1)), axis = 1)
			self.sigma_svd = np.append(self.sigma_svd, sig_tild[j])

	def disp_eig(self):
		for i in xrange(self.sub_s.shape[1]):
			sub_s = self.sub_s[:,i].reshape(self.mu_data.shape)
			temp = sub_s #+ self.mu_data)/255.0
			disp = temp.reshape(self.n,self.m)
			disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
			#stack = np.dstack((stack,disp))
			cv2.imshow('disp', disp)
			cv2.imshow('mean', self.mu_data.reshape((self.n, self.m))/255.0)
			cv2.waitKey(0)

	#Occlusion handling
	def weight_mask(self, frame):
		u = np.sum([self.xt_1[i].u*self.xt_1[i].wt for i in xrange(self.num)])
		v = np.sum([self.xt_1[i].v*self.xt_1[i].wt for i in xrange(self.num)])
		It = 0
		D = np.zeros(self.weightmask.shape)
		#need to make It as a mn cross k matrix
                It =frame[int(round(v - self.n/2.0,0)): int(round(v + self.n/2.0,0)), int(round(u - self.m/2.0,0)): int(round(u+self.m/2.0,0))]
		It = It.flatten()
		prod = It - np.matmul(np.matmul(self.sub_s, self.sub_s.T),It)
		#prod = prod.flatten()
		for i in xrange(prod.size):
			D[i] = prod[i]*self.weightmask[i]
			self.weightmask[i] = np.exp(-1*D[i]**2/self.sigma_wm**2)


	#some kind of cubic regression in temporal domain, 
	#predicts next point given the motion history
	#needs slight tweaks, slightly unstable model

	#OPEN TO SUGGESTIONS!!!! :P
	#Run and see
	def regress(self):
		#print 'u(t) = ', self.mean_best_ten[0]
		#print 'v(t) = ', self.mean_best_ten[1]

		self.prev_us = np.append(self.prev_us, np.ones((1,1))*self.mean_best_ten[0], axis = 0)		
		self.prev_vs = np.append(self.prev_vs, np.ones((1,1))*self.mean_best_ten[1], axis = 0)

		if(self.frames_passed >= self.n_0):
			self.prev_us = np.delete(self.prev_us, 0, axis = 0)
			self.prev_vs = np.delete(self.prev_vs, 0, axis = 0)

		if self.frames_passed == 0:
			t = np.zeros((1,4))
			t[0,0] = 1.
			self.t_matrix = np.append(self.t_matrix, t, axis = 0)
			self.t_poly_weights[:,0] = np.array([self.prev_us[0],0,0,0])
			self.t_poly_weights[:,1] = np.array([self.prev_vs[0],0,0,0])

		elif self.frames_passed == 1:
			t = np.array([(self.frames_passed**i) for i in range(4)]).reshape((1,4))	
			self.t_matrix = np.append(self.t_matrix, t, axis = 0)
			self.t_poly_weights[:,0] = np.array([self.prev_us[0], self.prev_us[1] - self.prev_us[0], 0, 0])
			self.t_poly_weights[:,1] = np.array([self.prev_vs[0], self.prev_vs[1] - self.prev_vs[0], 0, 0])
		
		elif self.frames_passed == 2:
			t = np.array([(self.frames_passed**i) for i in range(4)]).reshape((1,4))	
			self.t_matrix = np.append(self.t_matrix, t, axis = 0)

			self.t_poly_weights[:,0] = np.array([self.prev_us[0], 
				(-self.prev_us[2]+4*self.prev_us[1]-3*self.prev_us[0])/2.0, 
				(self.prev_us[2]-2*self.prev_us[1]+self.prev_us[0])/2.0, 0])
			
			self.t_poly_weights[:,1] = np.array([self.prev_vs[0], 
				(-self.prev_vs[2]+4*self.prev_vs[1]-3*self.prev_vs[0])/2.0, 
				(self.prev_vs[2]-2*self.prev_vs[1]+self.prev_vs[0])/2.0, 0])

		else:
			if self.frames_passed < self.n_0:
				t = np.array([(self.frames_passed**i)for i in range(4)]).reshape((1,4))	
				self.t_matrix = np.append(self.t_matrix, t, axis = 0)
			ata = np.dot(self.t_matrix.T, self.t_matrix)
			self.t_poly_weights[:,0] = np.dot(np.linalg.inv(ata), np.dot(self.t_matrix.T, self.prev_us)).reshape((4,))
			self.t_poly_weights[:,1] = np.dot(np.linalg.inv(ata), np.dot(self.t_matrix.T, self.prev_vs)).reshape((4,))


	def get_new_u(self):
		if(self.frames_passed >= self.n_0):
			tplusone = np.array([((self.n_0 + 0.5)**i) for i in range(4)])
		else:
			tplusone = np.array([((self.frames_passed - 0.5)**i) for i in range(4)])
		new_u = np.dot(tplusone, self.t_poly_weights[:,0].reshape((4,1)))[0]
		return new_u

	def get_new_v(self):
		if(self.frames_passed >= self.n_0):
			tplusone = np.array([((self.n_0)**i) for i in range(4)])
		else:
			tplusone = np.array([((self.frames_passed - 0.5)**i) for i in range(4)])
		new_v = np.dot(tplusone, self.t_poly_weights[:,1].reshape((4,1)))[0]
		return new_v

	def print_vel(self):
		t = (self.n_0 + 0.5)
		vel_u = self.t_poly_weights[1,0] + 2*self.t_poly_weights[2,0]*t + 3*self.t_poly_weights[3,0]*(t**2)
		vel_v = self.t_poly_weights[1,1] + 2*self.t_poly_weights[2,1]*t + 3*self.t_poly_weights[3,1]*(t**2)  

