import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from tqdm import tqdm

EPS = 1e-5
INF = 1e12

class IUSVMClassifier:

    def __init__(self, kernel, C):
        self.C = C
        self.kernel = kernel

        self.x = []
        self.y = []

        self.mu = np.array([0], np.float32)
        self.alphas = []

        self.S = []
        self.E = []
        self.O = []

        self.Q = np.array([0], np.float32)
        self.g = []


    def __call__(self, x):
        res = self.mu[0]

        for i, _ in enumerate(self.alphas):
            res += self.alphas[i] * self.y[i] * self.kernel(self.x[i], x)
        
        return res
    

    def calculateMaxAllowedDeltaAlpha(self, betas, gammas):
        k1 = None
        k2 = None
        k3 = None

        deltaAmax = []
        for i in range(1, len(self.S) + 1):
            if betas[i] > EPS:
                deltaAmax.append((self.C - self.alphas[self.S[i - 1]]) / betas[i])
            elif betas[i] < -EPS:
                deltaAmax.append(-self.alphas[self.S[i - 1]] / betas[i])
            else:
                deltaAmax.append(INF)

        if len(deltaAmax) != 0:
            i = np.argmin(deltaAmax)
            sign = 1 if deltaAmax[i] >= 0 else 0
            newArr = [abs(x_i) * sign for x_i in deltaAmax]
            k1 = self.S[np.argmin(newArr)]
            deltaAcS = np.min(newArr)
        else:
            deltaAcS = INF

        deltas = []
        for i in range(len(self.E)):
            if gammas[i + 1] > EPS:
                deltas.append(-self.g[self.E[i]] / gammas[i + 1])
            else:
                deltas.append(INF)
        shift = len(self.E)
        for i in range(len(self.O)):
            if gammas[i + 1 + shift] < -EPS:
                deltas.append(-self.g[self.O[i]] / gammas[i + 1 + shift])
            else:
                deltas.append(INF)

        if len(deltas) != 0:
            deltaAcR = min(deltas)
            i = np.argmin(deltas)
            if i < shift:
                k2 = self.E[i]
            else:
                k2 = self.O[i - shift]
        else:
            deltaAcR = INF

        if gammas[0] > EPS:
            deltaAcg = -self.g[-1] / gammas[0]
        else:
            deltaAcg = INF

        deltaAca = self.C - self.alphas[-1]
        k3 = len(self.alphas) - 1

        deltaAmaxC = min([deltaAcS, deltaAcR, deltaAcg, deltaAca])
        if deltaAmaxC == deltaAcS:
            return deltaAmaxC, k1
        elif deltaAmaxC == deltaAcR:
            return deltaAmaxC, k2
        else:
            return deltaAmaxC, k3
    

    def calculateBetas(self):
        nu = np.array([self.y[-1]] +
                      [self.y[s] * self.y[-1] * self.kernel(self.x[-1], self.x[s]) for s in self.S],
                      np.float32).reshape((-1, 1))
        return -1 * (self.Q @ nu).reshape((-1))


    def calculateGammas(self, betas):
        K_cc_cr = np.array([self.y[r] * self.y[-1] * self.kernel(self.x[-1], self.x[r]) for r in [-1] + self.E + self.O],
                           np.float32)
        m = np.concatenate([
            [[self.y[r]] for r in [-1] + self.E + self.O],
            [[self.y[r] * self.y[s] * self.kernel(self.x[r], self.x[s]) for s in self.S] for r in [-1] + self.E + self.O]],
            axis=1,
            dtype=np.float32
        )
        return m @ betas + K_cc_cr


    def addDataPair(self, x_c, y_c):
        alpha_c = 0
        g_c = y_c * self(x_c) - (1 + EPS)

        self.x.append(x_c)
        self.y.append(y_c)
        self.alphas.append(alpha_c)
        self.g.append(g_c)
        
        while self.g[-1] < -EPS and self.alphas[-1] < self.C:
            betas = self.calculateBetas()
            gammas = self.calculateGammas(betas)

            delta_alpha_c, k = self.calculateMaxAllowedDeltaAlpha(betas, gammas)

            if delta_alpha_c == 0:
                if len(self.S) == 0:
                    self.processEmptySetS()
                    continue

            self.mu += betas[0] * delta_alpha_c
            for i, _ in enumerate(self.S):
                self.alphas[self.S[i]] += betas[i + 1] * delta_alpha_c
            self.alphas[-1] += delta_alpha_c
            for i, _ in enumerate(self.E):
                self.g[self.E[i]] += gammas[i + 1] * delta_alpha_c
            shift = len(self.E)
            for i, _ in enumerate(self.O):
                self.g[self.O[i]] += gammas[i + 1 + shift] * delta_alpha_c
            self.g[-1] += gammas[0] * delta_alpha_c

            if k in self.S:
                self.deleteFromSetS(k)
                if abs(self.alphas[k]) < EPS:
                    self.O.append(k)
                else:
                    self.E.append(k)
            elif k in self.O:
                self.addToSetS(k)
                self.O.remove(k)
            elif k in self.E:
                self.addToSetS(k)
                self.E.remove(k)
            else:
                if abs(self.g[-1]) < EPS:
                    self.addToSetS(k)
                else:
                    self.E.append(k)
                break

        new_index = len(self.x) - 1
        if new_index not in self.S and new_index not in self.E and new_index not in self.O:
            self.O.append(new_index)


    def addToSetS(self, index):
        if len(self.S) == 0:
            self.Q = np.array([[-self.y[index] * self.y[index] * self.kernel(self.x[index], self.x[index]), self.y[index]],[self.y[index], 0]], np.float32)
            self.S.append(index)
            #print(self.Q)
            return
        nu = np.array([[self.y[index]] +
                      [self.y[s] * self.y[index] * self.kernel(self.x[index], self.x[s]) for s in self.S]],
                      np.float32).T
        #print(nu)
        betas = np.append(-1 * self.Q @ nu, [[1]], 0)
        #print(betas)
        k = self.y[index] * self.y[index] * self.kernel(self.x[index], self.x[index]) - nu.T @ self.Q @ nu
        #print(k)
        self.Q = np.append(
            np.append(self.Q, np.array([[0] for i in range(len(self.S) + 1)]), 1),
            np.array([[0 for i in range(len(self.S) + 2)]]),
            0
        ) + 1 / k * betas @ betas.T
        #print(self.Q)
        self.S.append(index)


    def processEmptySetS(self):
        for e in self.E:
            if abs(self.g[e]) < EPS:
                self.addToSetS(e)
                self.E.remove(e)
                return

        for o in self.O:
            if abs(self.g[o]) < EPS:
                self.addToSetS(o)
                self.O.remove(o)
                return

        mus_r = []
        for i in range(len(self.alphas) - 1):
            if i in self.E and -(self.y[i] / self.y[-1]) > EPS:
                mus_r.append(-self.g[i] / self.y[i])
            elif i in self.O and -(self.y[i] / self.y[-1]) < -EPS:
                mus_r.append(-self.g[i] / self.y[i])
        if len(mus_r) != 0:
            mu_r = min(mus_r)
        else:
            mu_r = INF
        mu_c = -self.g[-1] / self.y[-1]
        mu = min(mu_r, mu_c)
         
        for i in range(len(self.alphas)):
            self.g[i] += self.y[i] * mu
        self.mu += mu

        for e in self.E:
            if abs(self.g[e]) < EPS:
                self.addToSetS(e)
                self.E.remove(e)
                return

        for o in self.O:
            if abs(self.g[o]) < EPS:
                self.addToSetS(o)
                self.O.remove(o)
                return

        self.addToSetS(len(self.alphas) - 1)


    def deleteFromSetS(self, index):
        if len(self.S) == 1:
            self.Q = np.array([[0]], np.float32)
            self.S.remove(index)
            return
        k = self.S.index(index) + 1
        q_12 = np.delete(self.Q[:,k].reshape(-1, 1), k, 0)
        q_21 = np.delete(self.Q[k,:].reshape(1, -1), k, 1)
        q_22 = self.Q[k,k]
        q_11 = np.delete(np.delete(self.Q, k, 0), k, 1)
        self.Q = q_11 - q_12 @ q_21 / q_22
        self.S.remove(index)

def linKernel(x1, x2):
    return np.dot(x1, x2)
def othKernel(x1, x2, sigma):
    return np.exp(-np.dot(x1 - x2, x1 - x2) / (2 * sigma*sigma))

size = 250
space = 0.5
C = 1e2
plane = np.array([1.0, 1.0])
plane /= np.sqrt(np.dot(plane,plane))
dir = np.array([-1/plane[0], 1/plane[1]])
dir /= np.sqrt(np.dot(dir,dir))
b = 10
disp = 3
dist = 4

sift = 0

mean1 = plane * (b - dist/2) + dir * sift
mean2 = plane * (b + dist/2) + dir * sift

sample1 = multivariate_normal.rvs(mean1, ((disp,0),(0,disp)), size)
sample2 = multivariate_normal.rvs(mean2, ((disp,0),(0,disp)), size)

for point in sample1:
    dist = point[0] * plane[0] + point[1] * plane[1]
    if  dist - b + space > 0:
        point[0] -= 2 * plane[0] * (dist - b + space)
        point[1] -= 2 * plane[1] * (dist - b + space)

for point in sample2:
    dist = point[0] * plane[0] + point[1] * plane[1]
    if  dist - b - space < 0:
        point[0] -= 2 * plane[0] * (dist - b - space)
        point[1] -= 2 * plane[1] * (dist - b - space)

classifier = IUSVMClassifier(linKernel, C)

sample = []
color = []
min_x = 1e12
max_x = -1e12
min_y = 1e12
max_y = -1e12
for k in range(size):
    min_x = min(min_x, sample1[k][0])
    max_x = max(max_x, sample1[k][0])
    min_y = min(min_y, sample1[k][1])
    max_y = max(max_y, sample1[k][1])
    min_x = min(min_x, sample2[k][0])
    max_x = max(max_x, sample2[k][0])
    min_y = min(min_y, sample2[k][1])
    max_y = max(max_y, sample2[k][1])
    sample += [sample1[k]]
    color += ["red"]
    sample += [sample2[k]]
    color += ["blue"]
    classifier.addDataPair(sample1[k], 1)
    classifier.addDataPair(sample2[k], -1)
plt.figure()
delta_x = (max_x - min_x) / 25
delta_y = (max_y - min_y) / 25

x = np.arange(min_x, max_x, delta_x)
y = np.arange(min_y, max_y, delta_y)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i, y_val in enumerate(tqdm(y)):
    for j, x_val in enumerate(x):
        Z[i,j] = classifier([x_val,y_val])

plt.contour(X,Y,Z, levels=[-1,0,1], colors=['grey','k','grey'])

sampleS = np.array([sample[i] for i in classifier.S])
colorS = [color[i] for i in classifier.S]
sampleO = np.array([sample[i] for i in classifier.O])
colorO = [color[i] for i in classifier.O]
sampleE = np.array([sample[i] for i in classifier.E])
colorE = [color[i] for i in classifier.E]

if len(sampleS) != 0:
    plt.scatter(sampleS[:,0], sampleS[:,1], c=colorS, marker="*", s=128)
if len(sampleO) != 0:
    plt.scatter(sampleO[:,0], sampleO[:,1], c=colorO, marker="+", s=24)
if len(sampleE) != 0:
    plt.scatter(sampleE[:,0], sampleE[:,1], c=colorE, marker=".", s=64)



miss = 0
for k in range(size):
    if classifier(sample1[k]) * 1 < 0:
        miss += 1
    if classifier(sample2[k]) * -1 < 0:
        miss += 1

print(miss / size*2)
plt.show()

size = 1000
C = 1e1
disp = 10
mult = 1.5
data = multivariate_normal.rvs((0,0), ((disp,0),(0,disp)), size)
rad = 4

sample1 = []
sample2 = []
sample = []
color = []

min_x = 1e12
max_x = -1e12
min_y = 1e12
max_y = -1e12

classifier = IUSVMClassifier(lambda x, y:othKernel(x,y,8), C)

for i, point in enumerate(data):
    min_x = min(min_x, point[0])
    max_x = max(max_x, point[0])
    min_y = min(min_y, point[1])
    max_y = max(max_y, point[1])
    if np.sqrt(np.dot(point, point)) > rad:
        data[i] *= [1, 1]
        data[i] *= mult
        point = data[i]
        sample1.append(point)
        sample.append(point)
        color.append("red")
    else:
        data[i] *= [1, 1]
        sample2.append(point)
        sample.append(point)
        color.append("blue")

for i, point in enumerate(data[:100]):
    if color[i] == "red":
        classifier.addDataPair(point, 1)
    else:
        classifier.addDataPair(point, -1)

plt.figure()

delta_x = (max_x - min_x) / 100
delta_y = (max_y - min_y) / 100

x = np.arange(min_x, max_x, delta_x)
y = np.arange(min_y, max_y, delta_y)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i, y_val in enumerate(tqdm(y)):
    for j, x_val in enumerate(x):
        Z[i,j] = classifier([x_val,y_val])

plt.contour(X,Y,Z, levels=[-1,0,1], colors=['grey','k','grey'])

sampleS = np.array([data[i] for i in classifier.S])
colorS = [color[i] for i in classifier.S]
sampleO = np.array([data[i] for i in classifier.O])
colorO = [color[i] for i in classifier.O]
sampleE = np.array([data[i] for i in classifier.E])
colorE = [color[i] for i in classifier.E]

testO = []
testE = []
colorTestO = []
colorTestE = []
for j, point in enumerate(data[100:]):
    i = 100 + j
    val = 0
    col = None
    if color[i] == 'red':
        col = 'm'
        val = 1
    if color[i] == 'blue':
        col = 'c'
        val = -1
    if classifier(point) * val < 0:
        testE.append(point)
        colorTestE.append(col)
    else:
        testO.append(point)
        colorTestO.append(col)
    
testO = np.array(testO)
testE = np.array(testE)
if len(sampleS) != 0:
    plt.scatter(sampleS[:,0], sampleS[:,1], c=colorS, marker="*", s=128)
if len(sampleO) != 0:
    plt.scatter(sampleO[:,0], sampleO[:,1], c=colorO, marker="+", s=24)
if len(sampleE) != 0:
    plt.scatter(sampleE[:,0], sampleE[:,1], c=colorE, marker=".", s=64)
if len(testO) != 0:
    plt.scatter(testO[:,0], testO[:,1], c=colorTestO, marker="+", s=24)
if len(testE) != 0:
    plt.scatter(testE[:,0], testE[:,1], c=colorTestE, marker=".", s=64)



miss = 0
for i, point in enumerate(data):
    val = 0
    if color[i] == 'red':
        val = 1
    if color[i] == 'blue':
        val = -1
    if classifier(point) * val < 0:
        miss += 1

print(miss / size)
plt.show()


