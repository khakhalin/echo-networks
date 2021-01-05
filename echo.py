import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

# print('agraph loaded')

def create(n=50, minCommunity=3, tauCommunity=1.2, tauDegree=1):

	# --- Arguments:
	# n - number of nodes
	# minCommunity - minimal (starting) community size
	# tauCommunity = 1.2       # SOmething like an exponential constant for communities (except that they aren't random)
	# tauDegree = 1            # Exponential constant for degree distribution

	# --- Constants:
	degreePreference = 1     # Preferential attachment to high-degree nodes
	internalPreference = 100 # Preferential attachment within communities, as opposed to between communities

	# Map communities
	commIndex = np.zeros(n)
	iCommunity = 0                      # Community allocation
	edgeLeft = 0                        # Left border of current community
	thisSize = minCommunity             # Current community size
	while edgeLeft<n:
		edgeRight = min(edgeLeft+thisSize,n) # Right border of current community
		commIndex[edgeLeft:edgeRight] = iCommunity
		iCommunity += 1
		edgeLeft += thisSize
		thisSize = int(np.ceil(thisSize*tauCommunity))
		
		
	# Preferential attachment algorithm with communities

	A = np.zeros((n,n))
	deg = np.zeros(n,dtype=np.int8)    # Keep track of current degree, for preferential attachment later
	A[0,1] = 1
	A[1,0] = 1
	deg[0] += 1
	deg[1] += 1
	for iNode in range(1,n):
		targetDegree = int(np.ceil(np.random.exponential(scale=tauDegree))) # Target degree for this cell
		if commIndex[iNode] != commIndex[iNode-1]: # New community started, link it to the previous one
			A[iNode,iNode-1] = 1
			A[iNode-1,iNode] = 1
			deg[iNode] = 1
			deg[iNode-1] = 0
			targetDegree += -1
		for iAttempt in range(targetDegree):
			prob = 1 + (commIndex==commIndex[iNode])*internalPreference + deg*degreePreference
			# Relative probability of attaching to every node
			prob[iNode] = 0                               # No self-connections
			prob[np.argwhere(A[iNode,:]>0)] = 0           # Don't try to reconnect if the connection is already there
			iNew = np.random.choice(n,1,p=prob/sum(prob)) # From <n options, take 1, according to probabilities p
			A[iNode,iNew] = 1
			A[iNew,iNode] = 1
			deg[iNode] += 1
			deg[iNew] += 1

	# plt.imshow(A, cmap="Greys")


	# Forest fire percolation asymmetrisator
	# (For now based on direct matrix multiplication, as we can probably afford it)

	flag = 1
	fire = np.zeros(n,dtype=np.int8)
	fire[int(np.floor(np.random.rand(1)*n))] = 1
	coal = np.zeros(n,dtype=np.int8)
	count = 0
	B = A
	while np.sum(coal)<n and count<1000:
		count += 1
		coal = np.minimum(1,coal+fire)               # Counting all nodes that were visited
		newfire = np.minimum(1,np.matmul(B,fire))    # Propagate fire
		burningEdges = np.minimum(1,np.matmul(np.diag(fire),B)) # Edges activated on this path
		burningEdges = burningEdges - burningEdges*np.transpose(burningEdges)*np.tril(np.ones(n)) 
		# (If both directions are activated, take the upper one)
		
		B = np.maximum(0,B-np.transpose(burningEdges)) # Remove edges running opposite to those that were activated
		fire = newfire
		if np.sum(fire)==0:
			fire[int(np.floor(np.random.rand(1)*n))] = 1 # If fire went out, reignite randomly    

	# plt.imshow(B, cmap="Greys")
	
	return B