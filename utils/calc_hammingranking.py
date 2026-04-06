import numpy as np
import torch

def calc_hammingDist(B1, B2):
    B1, B2 = B1.to(torch.float32).cuda(), B2.to(torch.float32).cuda()
    q = B2.shape[1]
    distH = 0.5 * (q - torch.matmul(B1, B2.T))  
    return distH  


def calc_map(qB, rB, query_L, retrieval_L):
    if isinstance(qB, np.ndarray):
        qB = torch.tensor(qB, dtype=torch.float32).cuda()
    if isinstance(rB, np.ndarray):
        rB = torch.tensor(rB, dtype=torch.float32).cuda()
    
    query_L, retrieval_L = query_L.to(torch.float32).cuda(), retrieval_L.to(torch.float32).cuda()
    num_query = query_L.shape[0]
    mAP = 0

    for i in range(num_query):
        gnd = (torch.matmul(query_L[i], retrieval_L.T) > 0).float()
        tsum = max(1, int(torch.sum(gnd).item()))  
        if tsum == 0:
            continue

        hamm = calc_hammingDist(qB[i], rB)  
        ind = torch.argsort(hamm)  
        gnd = gnd[ind]

        count = torch.linspace(1, tsum, tsum, device='cuda')
        tindex = (torch.where(gnd == 1)[0] + 1).float()

        mAP += torch.mean(count / tindex)

    return mAP / num_query

if __name__=='__main__':
	qB = np.array([[ 1,-1, 1, 1],
								 [-1,-1,-1, 1],
								 [ 1, 1,-1, 1],
								 [ 1, 1, 1,-1]])
	rB = np.array([[ 1,-1, 1,-1],
								 [-1,-1, 1,-1],
								 [-1,-1, 1,-1],
								 [ 1, 1,-1,-1],
								 [-1, 1,-1,-1],
								 [ 1, 1,-1, 1]])
	query_L = np.array([[0, 1, 0, 0],
											[1, 1, 0, 0],
											[1, 0, 0, 1],
											[0, 1, 0, 1]])
	retrieval_L = np.array([[1, 0, 0, 1],
													[1, 1, 0, 0],
													[0, 1, 1, 0],
													[0, 0, 1, 0],
													[1, 0, 0, 0],
													[0, 0, 1, 0]])

	map = calc_map(qB, rB, query_L, retrieval_L)
	print(map)
