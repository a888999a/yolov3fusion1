import os
import shutil 

path_input=r"C:\Users\gary\Desktop\00\00\pic_90"
path_output=r"C:\Users\gary\Desktop\00\00\pick"
index=[4775,9096,12907,16547,18189,19826,28682]  #片段

for i in range(1,len(index)):
    out_1=os.path.join(path_output,str(i))
    for j in range(index[i]-300,index[i]+300):
        ori=os.path.join(path_input,"%06d"+".jpg")%j
        out_2=os.path.join(out_1,"%06d"+".jpg")%j
        shutil.copyfile(ori, out_2) 