## test
from PIL import Image 
from VesselSeg import VesselSeg_process
# img=Image.open('./test_imge/1.jpg')
# img.show()
# for i in range(3,10):
#     processer=VesselSeg_process(48,48,threshold=i/10)
#     res=processer(img)*255
#     res=Image.fromarray(res).convert("RGB")
#     res.save('./tmpsave/{}.jpg'.format(i))
#     raise
# print("finish")
import os
PATH="../autodl-tmp"
for sub_class in os.listdir(os.path.join(PATH,"test")):
    sub_class_number=len(os.listdir(os.path.join(PATH,'test',sub_class)))
    print("{} : {}".format(sub_class,sub_class_number))