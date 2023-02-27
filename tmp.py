## test
from PIL import Image 
from VesselSeg import VesselSeg_process
img=Image.open('./test_imge/1.jpg')
# img.show()
processer=VesselSeg_process(96,96)
res=processer(img)*255
res=Image.fromarray(res).convert("RGB")
res.save('tmp.jpg')