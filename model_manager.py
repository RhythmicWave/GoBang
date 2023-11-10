import os
import random
import torch

class ModelManager:
    def __init__(self,save_path="checkpoints",max_model_count=25):
        self.save_path=save_path
        self.model_list=[]
        self.max_model_count=max_model_count
        self.version=0

        self.model_list=self.get_all_models()

    def get_all_models(self):
        model_files = []
        for file in os.listdir(self.save_path):
            if file.endswith(".pth") and "last" not in file:
                model_files.append(file)

        model_files=sorted(model_files,key=lambda x:int(x.split(".")[0].split("_")[-1]))
        if len(model_files)>0:
            self.version=int(model_files[-1].split(".")[0].split("_")[-1])
        return model_files

    def save_model(self,state_dict):
        self.version+=1
        model_name = f"model_{self.version}.pth"
        if len(self.model_list) > self.max_model_count:
            remove_model = self.model_list[0]
            self.model_list.remove(remove_model)
            os.remove(os.path.join(self.save_path, remove_model))
        self.model_list.append(model_name)
        torch.save(state_dict,os.path.join(self.save_path,model_name))
        print(f"save model {model_name} successful!")


    def get_random_model(self,last_model_prob=0.8):
        is_last_model=True
        if len(self.model_list)==0:
            return None,is_last_model
        if random.random()>=last_model_prob:
            index=random.randint(0,len(self.model_list)-1)
            is_last_model=False
        else:
            index=len(self.model_list)-1
        return self.model_list[index],is_last_model

    def change_last_model(self):
        last_model=self.model_list[-1]
        if last_model is not None:
            filename=os.path.join(self.save_path,last_model)
            os.rename(filename,os.path.join(self.save_path,"last_model.pth"))
