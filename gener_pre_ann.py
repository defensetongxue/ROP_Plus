import os
import json
def generate_preplus_map(file_dic):
    preplus_map=[]
    file_list=os.listdir(file_dic)
    for i in file_list:
        if not i.endswith('.json'):
            continue
        with open(os.path.join(file_dic,i),'r') as f:
            data_list=json.load(f)
        for data in data_list:
            if data["plus_number"]>0 or data["pre_plus_number"]>0:
                preplus_map.append(data["image_name"])
    print(len(preplus_map))
    return preplus_map
def gen_plus_annotation(data_path,preplus_map):
    splits=['train','val','test']
    os.makedirs(os.path.join(data_path,'annotations_pre'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(data_path,'annotations_pre')}/*")
    for split in splits:
        with open(os.path.join(data_path,'annotations',f"{split}.json"),'r') as f:
            data_file=json.load(f)
        annotation_res=[]
        pos_cnt=0
        zero_cnt=0
        for data in data_file:
            if os.path.exists(os.path.join(data_path,'crop_optic_disc',data['image_name'])) :
                if data['image_name'] in preplus_map:
                    pos_cnt+=1
                    annotation_res.append({
                        "image_path":os.path.join(data_path,'crop_optic_disc',data['image_name']),
                        "image_name":data["image_name"],
                        "class":1
                    })
                else:
                    zero_cnt+=1
                    annotation_res.append({
                        "image_path":os.path.join(data_path,'crop_optic_disc',data['image_name']),
                        "image_name":data["image_name"],
                        "class":0
                    })
        print(f"Pre Plus {split}: total: {pos_cnt+zero_cnt} 1: {pos_cnt}, 0: {zero_cnt}")
        with open(os.path.join(data_path,'annotations_pre',f"{split}.json"),'w') as f:
            json.dump(annotation_res,f)
        
if __name__=='__main__':
    from config import get_config
    args=get_config()
    pre_map=generate_preplus_map(os.path.join(args.path_tar,'ridge'))
    gen_plus_annotation(args.path_tar,pre_map)


    
    