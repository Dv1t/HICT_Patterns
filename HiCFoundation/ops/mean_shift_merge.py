import numpy as np
from numba import jit
import math
@jit(nogil=True,nopython=True)
def carry_shift(point_cd,cnt,fmaxd,fsiv,xdim,ydim,dens):
    "2d density version"
    if True:
        point_dens=np.zeros(cnt)
        for i in range(cnt):
            if i%1000==0:
                print("mean shift",i,"/",cnt)
            #print(i)
            #print('start shifting for '+str(i))
            pos=np.zeros(2)
            for j in range(2):
                pos[j]=point_cd[i][j]
            if True:
                while True:
                    stp=np.zeros(2)
                    endp=np.zeros(2)
                    for j in range(2):
                        stp[j]=int(pos[j]-fmaxd)
                        if stp[j]<0:
                            stp[j]=0
                        endp[j]=int(pos[j]+fmaxd+1)
                    if endp[0]>=xdim:
                        endp[0]=xdim
                    if endp[1]>=ydim:
                        endp[1]=ydim
                    dtotal=0
                    pos2=np.zeros(2)
                    for xp in range(int(stp[0]),int(endp[0])):
                        rx=float(xp-pos[0])**2
                        for yp in range(int(stp[1]),int(endp[1])):
                            ry=float(yp-pos[1])**2

                            d2=rx+ry
                            v=np.exp(-1.5*d2*fsiv)*dens[xp,yp]#This is the bottom part of the equation, where pos represents y, (xp,yp,zp) represents xi
                            dtotal+=v
                            if v>0:
                                pos2[0]+=v*(float)(xp)#pos2 is for the top part of the equation
                                pos2[1]+=v*(float)(yp)

                    if dtotal==0:
                        break
                    rd=1.00/float(dtotal)
                    tempcd=np.zeros(2)
                    for j in range(2):
                        pos2[j]*=rd#Now we get the equation result
                        tempcd[j]=pos[j]-pos2[j]
                        pos[j]=pos2[j]#Prepare for iteration
                    check_d=tempcd[0]**2+tempcd[1]**2#Iteration until you find the place is stable
                    if check_d<0.001:
                        break

            for j in range(2):

                point_cd[i][j]=pos[j]
            point_dens[i]=dtotal/cnt
        return point_cd,point_dens
@jit(nogil=True,nopython=True)
def acc_merge_point(Ncd,dens,dmin,rv_range,rdcut,stock,cd,d2cut,member):
    if True:
        for i in range(Ncd-1):
            if i%10000==0:
                print(i)
            tmp=np.zeros(2)
            if (dens[i]-dmin)*rv_range < rdcut:
                stock[i]=0#Label the small density parts as unused parts
            if stock[i]==0:
                continue
            for j in range(i+1,Ncd):
                if stock[j]==0:
                    continue
                d2=0
                for k in range(2):
                    tmp[k]=cd[i][k]-cd[j][k]
                    d2+=tmp[k]**2
                if d2<d2cut:
                    #Mark the merged points to where it goes
                    if dens[i]>dens[j]:
                        stock[j]=0
                        member[j]=i
                    else:
                        stock[i]=0
                        member[i]=j
                        break#jump out of the second rotation, since i has been merged
        #Update member data, to updata some son/grandson points to original father point
        for i in range(Ncd):
            now=int(member[i])
            while now!=member[now]:#If it's not merged points, it will totates to find the father point(merged point)
                now=int(member[now])
            member[i]=now
        return stock,member
    
@jit(nogil=True,nopython=True)
def further_merge_point(count_loc, new_density, stock,
                                            new_location, d2cut,member):
    #before_count = len(np.argwhere(stock==1))
    for i in range(count_loc-1):
        if i%10000==0:
            print(i)
        member_i= member[i]
        for j in range(i+1,count_loc):
            member_j = member[j]
            if member_i==member_j:
                continue
            d2=0
            for k in range(2):
                d2+=(new_location[i][k]-new_location[j][k])**2
            if d2<d2cut:
                member_i_cluster_dens = new_density[member_i]
                member_j_cluster_dens = new_density[member_j]
                if member_i_cluster_dens>member_j_cluster_dens:
                    all_other_index = member==member_j
                    stock[all_other_index]=0
                    member[all_other_index]=member_i
                else:
                    all_other_index = member==member_i
                    stock[all_other_index]=0
                    member[all_other_index]=member_j
    #after_count = len(np.argwhere(stock==1))
    #print("after further merge, before count: %d, after count: %d"%(before_count,after_count))
    return stock,member


def mean_shift_merge(predict_array,cutoff=0.1):
    #generate mean shift detections
    #cutoff=0.1
    bandwidth = 2
    gstep = 1
    fs = (bandwidth / gstep) * 0.5
    fs = fs * fs
    fsiv = 1 / (float(fs))
    fmaxd = (bandwidth / gstep) * 2.0
    if cutoff==1:
        location = np.argwhere(predict_array>=cutoff)
    else:
        location = np.argwhere(predict_array>cutoff)
    count_loc = len(location)
    tmp_xdim=predict_array.shape[0]
    tmp_ydim = predict_array.shape[1]
    
    new_location, new_density = carry_shift(location, count_loc, fmaxd, fsiv,
                                            tmp_xdim, tmp_ydim, predict_array)
    if len(new_location)==0 or len(new_density)==0:
        return []
    dmin = np.min(new_density)
    dmax = np.max(new_density)
    drange = dmax - dmin
    print('here we get the density range %f' % drange)
    rv_range = 1.0 / drange
    rdcut = 0.01
    stock  = np.ones(count_loc)
    d2cut = 2 ** 2 #important merge criteria
    member = np.arange(count_loc)
    
    stock, member = acc_merge_point(count_loc, new_density, dmin,
                                                  rv_range, rdcut, stock,
                                                  new_location, d2cut,member)
    #further merge, if we observe any two subpoints are close than 2, then we still merge them.
    stock, member = further_merge_point(count_loc, new_density, stock,
                                            new_location, d2cut,member)
    final_loc_list=[]
    for i in range(count_loc):
        if stock[i] == 1:
            final_loc_list.append(new_location[i])
    final_loc_list =np.stack(final_loc_list,axis=0)
    return final_loc_list