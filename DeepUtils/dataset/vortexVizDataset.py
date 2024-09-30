from .build import DATASETS
from .data_utils import *
import torch,os, time,tqdm
from PIL import Image

def read_root_metaJson(root_directory):
    try_path0=os.path.join(root_directory, 'meta.json')
    try_path1=os.path.join(root_directory, 'train/meta.json')
    if os.path.exists(try_path0):
        return read_json_file(try_path0)
    elif os.path.exists(try_path1):
        return  read_json_file(try_path1)
        
    else:
        raise ValueError("no root meta file found.")
def visualize_binary_image(binary_image, sub_folder, file_name, line_number):

    # Create a directory for visualizations if it doesn't exist
    viz_dir = sub_folder
    os.makedirs(viz_dir, exist_ok=True)
    # Generate a unique filename for the PNG
    base_name =file_name
    png_filename = f"{base_name}_line{line_number}.png"
    png_path = os.path.join(viz_dir, png_filename)

    pil_image = Image.fromarray((binary_image * 255).astype('uint8'), 'L')
    pil_image.save(png_path)
    print(f"Saved visualization: {png_path}")   
    
@DATASETS.register_module()
class VortexVizDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,split, transform,**kwargs):
        self.directory_path=data_dir
        self.dataName=[]
        self.data=[]
        self.label=[]
        self.dastasetMetaInfo={}
        self.split=split
        self.preLoading(split)
        self.transform = transform
  
    def getSampleName(self,sampleIdx):        
        return self.dataName[sampleIdx]   
     
    def preLoading(self,mode):                
 
        print(f"Preloading [{mode}] data......")
        start = time.time()         
        #self.directory_path should have meta.json
        self.dastasetMetaInfo=read_root_metaJson(self.directory_path)
        
        #train_directory_path should be  the direct parent folder of all "rc_xxxx_n_xxx"  folders
        split_str =mode if mode!="val" else "validation"
        target_directory_path=os.path.join(self.directory_path,split_str) 
        if  os.path.exists(target_directory_path)==False and mode=="val":
            target_directory_path=os.path.join(self.directory_path,"test") 
            print(f"Didn't find validation folder, use {target_directory_path} folder as [{mode}] data......")
        if  os.path.exists(target_directory_path)==False and mode=="test":
            target_directory_path=os.path.join(self.directory_path,"validation") 
            print(f"Didn't find test folder, use {target_directory_path} folder as [{mode}] data......")
        
        rc_n_subfoders=[os.path.join(target_directory_path,folder) for folder in os.listdir(target_directory_path) if os.path.isdir(os.path.join(target_directory_path, folder))]
        for folder_name in tqdm.tqdm(rc_n_subfoders) :           
                self.loadOneVortexVizFolder(folder_name)          
        #logging dataset information    and time consuming of preloading data
        elapsed = time.time()
        elapsed = elapsed - start
        print(f"Preloading  [{mode}] data......done")
        print(f"Total number of [{mode}] data:{len(self.data)}, took {elapsed} seconds")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data=self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data , self.label[idx]
    

    def loadOneVortexVizFolder(self,sub_folder:str):
        #Xdim,Ydim is the binary Image size
        Xdim,Ydim=self.dastasetMetaInfo["Xdim"],self.dastasetMetaInfo["Ydim"]
        maximumPathlineLengthVortexViz=self.dastasetMetaInfo["maximumPathlineLengthVortexViz"]
        #find all *.bin data in this subfoder
        inforVectorFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin') and "CulmulateAbsCurl" in f]
        for File in inforVectorFiles:
            inforVectorBinPath=os.path.join(sub_folder,File)
            pathlineBinaryImagePath=inforVectorBinPath.replace("_CulmulateAbsCurl.bin","pathlineImages.bin")
            labelBinaryImagePath=inforVectorBinPath.replace("_CulmulateAbsCurl.bin","_segmentationLabel.bin")
            raw_Binary = read_binary_file(pathlineBinaryImagePath,dtype=np.uint8).astype(np.float32)
            raw_Binary=raw_Binary.reshape(-1,Ydim,Xdim)
            pathlinesCount=raw_Binary.shape[0]
            label= read_binary_file(labelBinaryImagePath,dtype=np.uint8).reshape(pathlinesCount).astype(np.float32)
            infor_Binary = read_binary_file(inforVectorBinPath,dtype=np.float32)
            infor_Binary=infor_Binary.reshape(pathlinesCount,maximumPathlineLengthVortexViz)
            for line in range(pathlinesCount):
                #shape of infor_Binary is  L
                #shape of pathlineBinaryImagePath is 64x64
                # self.data.append( (torch.tensor(raw_Binary[line]).unsqueeze(1), torch.tensor(infor_Binary[line])) 
                # )
                # self.data.append( (torch.tensor(raw_Binary[line]).unsqueeze(1), torch.tensor(infor_Binary[line])) )
                tensor_Image=torch.tensor(raw_Binary[line])
                tensor_Info=torch.tensor(infor_Binary[line])
                merged_tensor = torch.cat([tensor_Image.flatten(), tensor_Info])
                self.data.append( merged_tensor)
                self.label.append(torch.tensor(label[line]))
                # if line % 11==0:
                #     label_name="inside" if label[line]>=0.5 else "out"
                #     visualize_binary_image(raw_Binary[line], "./temp/", keep_path_last_n_names(pathlineBinaryImagePath,2).replace(".bin",label_name), line)
                    
            
      