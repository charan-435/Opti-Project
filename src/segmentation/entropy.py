import cv2
import numpy as np
import os

# shannon entropy logic
def calc_entropy(p):
    p = p[p > 0]                       #P vector --> Fraction of each gray scale intensity 
    return -np.sum(p * np.log2(p))

#<=======================================================>
def get_multi_entropy(h_norm, t_list):
    t_list = sorted([int(t) for t in t_list])
    regs = []       #Contains the 4 vectors for each region 
    
    last = 0
    for t in t_list:               #filling regions(regs)
        regs.append(h_norm[last:t])
        last = t
    regs.append(h_norm[last:])

    ent = 0                #calculating entropy for the given t_list
    for r in regs:
        p_sum = r.sum()
        if p_sum > 0:
            ent += calc_entropy(r / p_sum)
    return ent
#returns total entropy for the given threshold vector
#<=======================================================>



class ThresholdFinder:
    
    
    def __init__(self, n=20, iters=50, k=3):
        self.n = n
        self.T = iters
        self.k = k
        self.min_v = 1
        self.max_v = 254
   # n =20 ----> rnadom objects
   # iters = 50 -----> iterations
   # k = 3 -----> 4 regions
#<=======================================================>
    def fitness(self, positions, h_norm):
        f = []
        for p in positions:
            t = np.clip(p, self.min_v, self.max_v)
            f.append(get_multi_entropy(h_norm, t))
        return np.array(f)
#returns the fitness for each object 
#<=======================================================>
    def find(self, img):
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
          #image type changing
        
        h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hn = h / h.sum()
         #NORMALISED HISTOGRAM
#<=======================================================>        
        r = np.random.default_rng()

        # init
        p = r.uniform(self.min_v, self.max_v, (self.n, self.k))
        #position matrix , 20*3 dimensions 
        
        d = r.random((self.n, self.k))
        #density for each object 
        v = r.random((self.n, self.k))
        #velocity for each object 
        a = r.uniform(self.min_v, self.max_v, (self.n, self.k))
        #acceleration for each object 
#<=======================================================>        
        fit = self.fitness(p, hn)
        #calculate initial fitness for each object , 20 size vector 
        best_i = np.argmax(fit)
        #best value (argument/which position has best entropy)
        x_best = p[best_i].copy()
#<=======================================================>

       
        #20 iterations 
        for t in range(1, self.T + 1):
            
            tf = np.exp((t - self.T) / self.T)
            #transfer function based on current iteration 
            #tells when to do exploration vs exploitation 
            #gradual increase due to exponential nature 
            
            
            
            #MOVEMENT SCALING FACTOR
            
            dist_val = max(np.exp((self.T - t) / self.T) - (t / self.T), 1e-8)
            #A VALUE TO DECIDE SCALING OF MOVEMENT ---> early iterations == fast movement
            # ELSE ----->late iterations == slow_movement

            #update densities and velocities towards best 
            d = d + r.random((self.n, self.k)) * (d[best_i] - d)
            v = v + r.random((self.n, self.k)) * (v[best_i] - v)

            if tf <= 0.5:
                # exploration
                m_idx = r.integers(0, self.n, self.n)  #RANDOM OBJECT SELECTED TO  MOVE OTHER OBJECTS
                a = (d[m_idx] * v[m_idx] * a[m_idx]) / (d * v + 1e-8)
            else:
                # exploitation
                a = (d[best_i] * v[best_i] * a[best_i]) / (d * v + 1e-8)   #BEST OBJECT SELECTED TO MOVE OTHER OBJECTS

            a_min, a_max = a.min(), a.max()
            an = (0.1 + 0.8 * (a - a_min) / (a_max - a_min) if a_max > a_min else np.full_like(a, 0.5))
            #1)NORMALIZING ACC
            #2)SCALING ACC FROM 0.1 to 0.9 {avoids extreme movements}
            #3)EDGE CASE --- > IF AMAX == AMIN (MAKE EVERYTHING NEUTRAL VALUE = 0.5) 
            
            
            if tf <= 0.5:  
                xr = r.uniform(self.min_v, self.max_v, (self.n, self.k)) #RANDOM POSITION FOR EACH OBJECT 
                p = p + 2 * r.random((self.n, self.k)) * an * dist_val * (xr - p)
                #MOVE TO THE CORRESPONDING RANDOM OBJECT WITH SOME MORE RANDOM PARAMETERS FOR EACH DIMENSION
            else:
                F = np.where(r.random((self.n, self.k)) > 0.5, 1, -1) #EACH VALUE EITHER +1 or -1
                #ENSURES MOVEMENT AROUND BOTH DIRECTION OF THE BEST SOLUTION AND NOT JUST ONE DIRECTION
                p = x_best + F * 6 * r.random((self.n, self.k)) * an * dist_val * (x_best - p)
              
            p = np.clip(p, self.min_v, self.max_v)  #filter P for the grayscale values 0->254
            fit = self.fitness(p, hn)  #new fitness 
            new_b = np.argmax(fit)     #new best object 

            if fit[new_b] > fit[best_i]:
                best_i = new_b
                x_best = p[new_b].copy()
             #if new best better keep it as best 
        return sorted([int(round(val)) for val in x_best])
        #finally return the best threshold


def do_segment(img, k=3):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t_vals = ThresholdFinder(k=k).find(img)
    lvls = [0] + t_vals + [256]  #final region vector 
    
    n_r = len(lvls) - 1
    ints = [int(i * 255 / (n_r - 1)) for i in range(n_r)]
    #assign new and uniform intensties to each region for differentiating 
    
    res = np.zeros_like(img)
    for i in range(n_r): #for each region 
        low, high = lvls[i], lvls[i+1]
        if i == n_r - 1:
            m = (img >= low) & (img <= high)   #find the indxs having the intensities in this region 
        else:
            m = (img >= low) & (img < high)
        res[m] = ints[i]
    #create the result image 

    return res, t_vals

def run_dataset_seg(in_dir, out_dir, labels=("yes", "no"), k=3):
    for l in labels:
        p_in = os.path.join(in_dir, l)
        p_out = os.path.join(out_dir, l)
        if not os.path.exists(p_out): os.makedirs(p_out)

        files = [f for f in os.listdir(p_in) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"Seg processing {l} - {len(files)} files")

        for i, f in enumerate(files, 1):
            img = cv2.imread(os.path.join(p_in, f), cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            seg, t = do_segment(img, k)
            out_f = os.path.splitext(f)[0] + ".png"
            cv2.imwrite(os.path.join(p_out, out_f), seg)
            
            if i % 20 == 0:
                print(f"{i}/{len(files)} - {t}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_dataset_seg(
        in_dir=os.path.join(base, "data/augmented"),
        out_dir=os.path.join(base, "data/segmented_multi"),
        labels=("yes", "no"),
        k=3
    )
    print("finished.")