import os
import scipy.io as sio    
from collections import Counter
from tqdm import tqdm  
import numpy as np    
import json

CELL_TYPE = [
    'neoplastic',
    'inflammatory',
    'connective',
    'dead',
    'epithelial'
]

if __name__ == "__main__":
    
    path_ = "./masks/"
    all_mat = sorted(os.listdir(path_))
    
    tissue_types = []
    for mm in all_mat:
        tissue_types.append(mm[mm.find("-")+1:-4])
    
    all_tissue = Counter(tissue_types)
    
    # for val in all_tissue:
    #     print(val)
    #     print("%.2f"%(all_tissue[val] / 5179 *100))
    
    tissue_type_all = [a for a in all_tissue]
    
    text_descrips = dict()
    
    total_num = 0
    main_cell = 0
    
    for every_tissue in tissue_type_all:

        ####### 每个组织类别的细胞数量统计
        cell_count= dict({0:[], 1:[], 2:[], 3:[], 4:[]})
        for mask__ in tqdm(all_mat):
            if every_tissue not in mask__:
                continue
            
            mm_ = sio.loadmat(os.path.join(path_, mask__))
            map_ = mm_['inst_map']
            
            for cell_type_ in [0, 1, 2, 3, 4]:
                dead_c = map_[:,:,cell_type_]
                unique, counts = np.unique(dead_c, return_counts=True)
                ###包含细胞
                if counts.shape[0] > 1:
                    cell_count[cell_type_].append( counts.shape[0]-1 )
                
        ##### top 30% and bottom 30%
        threshold_top = []
        threshold_bottom = []
        
        for cell_t in cell_count:
            cell_num = sorted(Counter(cell_count[cell_t]))
            if len(cell_num) == 0:
                cell_num = [0]
            index_t = int(0.7 * len(cell_num))
            threshold_top.append( cell_num[index_t] )
            index_b = int(0.3 * len(cell_num))
            threshold_bottom.append( cell_num[index_b] )                   
        
    
                
        ####### 确定组织类型和细胞分布
        path_test = "F:/Patholog-related/test/masks/"
        test_mat = sorted(os.listdir(path_test))
        for mask__ in tqdm( all_mat+test_mat ):
            if every_tissue not in mask__:
                continue
            
            if 'train' in mask__:
                mm_ = sio.loadmat(os.path.join(path_, mask__))
            else:
                mm_ = sio.loadmat(os.path.join(path_test, mask__))
                
            map_ = mm_['inst_map']
            
            unique_all, count_all = np.unique(map_[:,:,:-1], return_counts=True)
            total_cell = count_all.shape[0]-1 
            
            cell_type_patches = dict()
            for cell_type_ in [0, 1, 2, 3, 4]:

                dead_c = map_[:,:,cell_type_]
                unique, counts = np.unique(dead_c, return_counts=True)
                ###包含细胞
                if counts.shape[0] > 1:
                    cell_type_patches[CELL_TYPE[cell_type_]] = counts.shape[0]-1
            
            #########确定组织类型 --------
            sorted_c = sorted(cell_type_patches.items(), key=lambda kv: kv[1], reverse=True)
            
            sorted_cell_type = []
            sorted_cell_count = []
            for sc_ in sorted_c:
                sorted_cell_type.append(sc_[0])
                sorted_cell_count.append(sc_[1])
                
                
            if 'neoplastic' in sorted_cell_type:
                tissue_descp = '%s tumor'%every_tissue
            else:
                if len(sorted_cell_type) % 2 == 0:
                    tissue_descp = 'normal %s'%every_tissue
                else:
                    tissue_descp = 'benign %s'%every_tissue  
            
            if  len(sorted_cell_count) == 0:  
                    if sc_[1] %2 == 0:      
                        text_description = "%s image of empty space, "%(tissue_descp)
                    else:
                        text_description = "%s image of background, "%(tissue_descp)
                    
            elif  sorted_cell_count[0] / total_cell  >= 0.5:
                    text_description = "%s image of %s tissue, "%(tissue_descp, sorted_cell_type[0])
            
            elif sorted_cell_count[0] / total_cell  > 0.3 and sorted_cell_count[1] / total_cell  > 0.3 and \
                sorted_cell_count[2] / total_cell  > 0.3 and total_cell > 15:
                    text_description = "%s image of %s, %s and %s tissue, "%(tissue_descp, sorted_cell_type[0]\
                        , sorted_cell_type[1], sorted_cell_type[2])

            elif sorted_cell_count[0] / total_cell  > 0.3 and sorted_cell_count[1] / total_cell  > 0.3 \
                 and total_cell > 10:
                    text_description = "%s image of %s and %s tissue, "%(tissue_descp, sorted_cell_type[0]\
                        , sorted_cell_type[1])    
            else:
                    text_description = "%s image of %s tissue, "%(tissue_descp, sorted_cell_type[0])   
            
            text_description = text_description.replace("dead", "necrostic")
            
            ########细胞数量分布描述 ----------
            for cell_tp, cell_num in zip(sorted_cell_type, sorted_cell_count):

                thres_top = threshold_top[CELL_TYPE.index(cell_tp)]
                thres_bot = threshold_bottom[CELL_TYPE.index(cell_tp)] 
                
                if cell_num >= 75:
                    text_description += "hundreds of %s, "%cell_tp
                                       
                elif cell_num >= thres_top:
                    text_description += "many %s, "%cell_tp  
                         
                elif cell_num <= thres_bot:
                    text_description += "few %s, "%cell_tp  

                else:
                    if cell_num % 2 == 0:
                        text_description += "some %s, "%cell_tp  
                    else:
                        text_description += "several %s, "%cell_tp                      

            text_descrips[mask__] = text_description[:-2] if text_description[-2:] == ", " else text_description
        
        with open('./PanNuke_text.txt', 'w', encoding='utf-8') as f:
            f.write(json.dumps(text_descrips))        