#include "particle.cpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <cmath>

using namespace Eigen;

class part_filt{
	int num;
	int n,m;
	int frames_passed;
	float forget_factor;
	float mean_best_10[2];
	ArrayXf mu_data;
	int k;
	vector<int, int> predicted;

	ArrayXXf sub_s;
	ArrayXXf sigma_svd; 

	vector<particle> xt;
	vector<particle> xt_1;

	int n_0;
	'''
	//WRITING THESE HERE JUST AS A REFERENCE
	//prev_us = np.zeros((0,1))
	//prev_vs = np.zeros((0,1))
	//t_poly_weights = np.zeros((4,2))
	//t_matrix = np.zeros((0,4))
	'''
	ArrayXf prev_us;
	ArrayXf prev_vs;
	ArrayXXf t_poly_weights;
	ArrayXf t_matrix;


	float sig_mse;
	float sig_d;
	float alpha;
	float sigma_wm;
	public:
		part_filt(int num, temp, int w, int h, float sig_d, float sig_mse, int init_center, float sigma_wm = 1, float ff = 0.9, int n_0 = 6, int k = 10, float alpha = 0.7){
			this.num = num;
			//n,m = temp.shape[:2]  //USED TO BE THIS WAY
			this.frames_passed = 0;
			this.forget_factor = ff;
			this.mu_data = //FILL THIS
			this.mean_best_10[0] = 0;	
			this.mean_best_10[1] = 0;

			this.k = k;
			this.sub_s = ArrayXf::zeros(n*m,0);
			this.sigma_svd = ArrayXXf::zeros(0,0);

			this.n_0 = n_0;
			this.prev_us = ArrayXXf::zeros(0,1)
			this.prev_vs = ArrayXXf::zeros(0,1)
			this.t_poly_weights = ArrayXXf::zeros(4,2)
			this.t_matrix = ArrayXXf::zeros(0,4)


			this.sig_mse = sig_mse;
			this.sig_d = sig_d;
			this.alpha = alpha;
			this.sigma_wm = sigma_wm;
			for(int i=0; i<num; i++){
				particle temp(init_center[0], init_center[1],1.0/num);
				this.xt_1.push_back(temp);
			}
		}

		void sample(ArrayXXf frame)
		{
			num = xt_1.size();
			int total_p =0, i=0;
			float eta = 0.0;

			float nlmz = 0, nlmz_u=0, nlmz_v=0;
			for(int i=0; i<10; i++)
			{
				nlmz += xt_1[i].wt;
				nlmz_u += xt_1[i].wt*xt_1[i].u;
				nlmz_v += xt_1[i].wt*xt_1[i].v;
			}
			this.mean_best_10[0] = nlmz_u/nlmz;
			this.mean_best_10[1] = nlmz_v/nlmz;

			this.regress()
			float u_t_plus_1 = this.get_new_u();
			float v_t_plus_1 = this.get_new_v();

			predicted.push_back(int(u_t_plus_1), int(v_t_plus_1)); //NOT SURE OF SYNTAX, PLS CHECK
			this.print_vel();

			int n_vel = int(this.alpha*this.num);
			while(i<n_vel)
			{
				int p = int(round(this.xt_1[i].wt*n_vel));
				total_p += p;
				if(total_p<n_vel)
				{
					vector<float> delt_u = //Somehow generate a vector of num elements p, mean u_t_plus_1 - self.mean_best_ten[0], sigma sigd
					vector<float> delt_v = //Somehow generate a vector of num elements p, mean v_t_plus_1 - self.mean_best_ten[1], sigma sigd
					int j=0;
					while(j<p)
					{
						new_u = this.xt_1[i].u + delt_u[j];
						new_v = this.xt_1[i].v + delt_v[j];
						new_wt = self.pzt(frame, new_u, new_v);
						eta+=new_wt;
						this.xt.push_back(particle(new_u, new_v, new_wt));
						j+=1;
					}
				}
				else
				{
					vector<float> delt_u = //Somehow generate a vector of num elements n_vel - total_p + p, mean u_t_plus_1 - self.mean_best_ten[0], sigma sigd
					vector<float> delt_v = //Somehow generate a vector of num elements n_vel - total_p + p, mean v_t_plus_1 - self.mean_best_ten[1], sigma sigd
					int j=0;
					while(j<n_vel - total_p + p)
					{
						new_u = this.xt_1[i].u + delt_u[j];
						new_v = this.xt_1[i].v + delt_v[j];
						new_wt = self.pzt(frame, new_u, new_v);
						eta+=new_wt;
						this.xt.push_back(particle(new_u, new_v, new_wt));
						j+=1;
					}
				}
				i++;
			}
			if(total_p < n_vel){
				vector<float> delt_u = //THIS AGAIN!! np.random.normal(u_t_plus_1 - self.mean_best_ten[0], self.sig_d, n_vel - total_p)
				//OR #delt_u = np.random.normal(u_t_plus_1 - self.xt_1[0].u, self.sig_d, n_vel - total_p)
				vector<float> delt_v = //np.random.normal(v_t_plus_1 - self.mean_best_ten[1], self.sig_d, n_vel - total_p)
				//OR #delt_v = np.random.normal(v_t_plus_1 - self.xt_1[0].v, self.sig_d, n_vel - total_p)
				j = 0
				while(j < n_vel - total_p)
				{
					new_u = self.xt_1[j].u + delt_u[j];
					new_v = self.xt_1[j].v + delt_v[j];
					new_wt = self.pzt(frame, new_u, new_v);
					eta+=new_wt;
					this.xt.push_back(particle(new_u, new_v, new_wt));
					j+=1;
				}
			}
			int no_wo_vel = self.num - n_vel;
			vector<float> delt_u = //np.random.normal(0, self.sig_d, n_wo_vel)
			vector<float> delt_v = //np.random.normal(0, self.sig_d, n_wo_vel)
			int i = 0;

			while (i < n_wo_vel){
				new_u = self.mean_best_ten[0] + delt_u[i];
				new_v = self.mean_best_ten[1] + delt_v[i];
				new_wt = self.pzt(frame, new_u, new_v);
				eta+=new_wt;
				this.xt.push_back(particle(new_u, new_v, new_wt));
				i+=1;
			}

			i = 0;
			while(i < self.num){
				this.xt[i].wt/=eta;
				i+=1;
			}
			
			this.sort_by_weight();	
			this.xt_1 = this.xt;
			this.xt.clear();//NOT SURE BOUT SYNTAX

			this.update_temp();
			this.frames_passed++;
		}

		float pzt(ArrayXXf frame, int u, int v)
		{
			int h,w = //get dimension of frame, idk syntax
			ArrayXXf img2; 
			if(u<=w-this.m/2 and u >= this.m/2 and v>= this.n/2 and v<=h-this.n/2)
			{
				//IF YOU CAN DO FOR 1, YOU CAN DO FOR ALL
				if(self.n%2==0 and self.m%2 == 0)
					img2 = //frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2)]
				else if(self.n%2==0 and self.m%2 != 0)
					img2 = //frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2 )+ 1]
				else if(self.n%2!=0 and self.m%2 == 0)
					img2 = //frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2)]
				else
					img2 = //frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2) + 1]

				img2 = //We need to convert to vector, use eigen library. img2.flatten()
				img2 = //same reason as above. img2.reshape((img2.size,1))

				//Real stuff happens here
				float err = this.MSE(img2) ;
				float weight = exp(-err/2/sig_mse/sig_mse);
				

				return weight
			}
			else
				return 0;
		}

		float MSE(ArrayXXf img2)
		{
			ArrayXf z = img2 - this.mu_data;
			ArrayXf p = this.sub_s*(this.sub_s.transpose()*z);
			ArrayXf l;
			float sum=0;
			for(int i=0; i<this.m*this.n; i++)
			{
				l[i] = (z[i]-p[i])*(z[i]-p[i]);
				sum += l[i]/(l[i]+3*38*38); //AGAIN, PLEASE CHECK THIS, I'M NOT SURE
			}
			return sum;
		}

		void sort_by_weight()
		{
			MergeSort m;
			m.sort(this.xt, 0, this.num -1);
		}

		void update_temp(ArrayXXf frame)
		{
			float nlmz = 0, nlmz_u=0, nlmz_v=0;
			for(int i=0; i<10; i++)
			{
				nlmz += xt_1[i].wt;
				nlmz_u += xt_1[i].wt*xt_1[i].u;
				nlmz_v += xt_1[i].wt*xt_1[i].v;
			}
			float u = nlmz_u/nlmz;
			float v = nlmz_v/nlmz;
			ArrayXXf img2;
			//SAME THNG AS BEFORE
			if(self.n%2==0 and self.m%2 == 0)
				img2 = //frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2)]
			else if(self.n%2==0 and self.m%2 != 0)
				img2 = //frame[int(v - self.n/2): int(v + self.n/2), int(u - self.m/2): int(u+self.m/2 )+ 1]
			else if(self.n%2!=0 and self.m%2 == 0)
				img2 = //frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2)]
			else
				img2 = //frame[int(v - self.n/2): int(v + self.n/2) +1, int(u - self.m/2): int(u+self.m/2) + 1]
		}


};