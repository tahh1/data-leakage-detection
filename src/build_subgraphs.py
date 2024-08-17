import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import networkx as nx
import re
import torch
import dgl
import os,sys
import re
import dgl
import networkx as nx
from dgl.data.utils import save_graphs
from utils import remove_files

def create_binary_features(lpd):
    
      labels =["LoadField","LoadIndex","StoreFieldSSA","StoreIndexSSA","StoreIndex","AssignVar",
              "InterProcAssign","AssignGlobal","LoadSlice","StoreSliceSSA","LocalMeth","NonLocalMethod",
              "Base","Field","Index","Start","End","Step","Var","Param","Value","Train","_Phi_",
              "Test","Left","Right","Instr", "TrainData","TestData", "AssignBoolConstant", "AssignStrConstant",
               "AssignIntConstant","AssignFloatConstant", "Input", "Return" , "LoadExtSlice" , "StoreExtSlice",
              "Add", "Sub", "Mult", "Div", "FloorDiv","Mod","Pow","BitAnd","BitOr","BitXor","LShift","RShift",
               "Invert","Not","UAdd", "USub"
               ]
      
      mlb = MultiLabelBinarizer(classes=list(map(lambda x:x.lower(),labels)))
      features = lpd.iloc[:,1].to_list()
      features = [set(labels.lower().split()) for labels in features]
      features = mlb.fit_transform(features)

      return torch.from_numpy(features).to(torch.float)



def extract_instr_and_vars(FlowVarTransformation,FlowVarStoreIndex):

    unique_instr = sorted(list(set(
                            read_index_or_empty(FlowVarTransformation,"InstructionId")
                            +read_index_or_empty(FlowVarStoreIndex,"InstructionId"))))
    index_mapping = {instr:index for index,instr in enumerate(unique_instr)}
    unique_vars = sorted(list(set(
                           read_index_or_empty(FlowVarTransformation,"ToId")+
                           read_index_or_empty(FlowVarTransformation,"FromId")+
                           read_index_or_empty(FlowVarStoreIndex,"ToId")+
                           read_index_or_empty(FlowVarStoreIndex,"FromId"))))
    unique_vars = list(filter(lambda x: re.fullmatch("\[(\$?invo\d+?)?, (\$?invo\d+?)?\]",x)==None,unique_vars))
    num_instr= len(index_mapping)
    for index,var in enumerate(unique_vars):
        index_mapping[var]=num_instr+index

    return unique_instr,unique_vars,index_mapping




def build_adj(flow_from_inst,flow_to_inst, unique_instr,unique_vars):

    adj= np.zeros((len(unique_instr)+len(unique_vars),len(unique_instr)+len(unique_vars)))
    for index,instr in enumerate(unique_instr):
        for to_vars in flow_from_inst[instr]:
            adj[index][index_mapping[to_vars]]=1
        for from_vars in flow_to_inst[instr]:
            adj[index_mapping[from_vars]][index]=1

    return adj


def get_columns(filename):
    d = {
        "FLowVarTransformation.csv": ['To', 'ToCtx', 'Instr', 'From','FromCtx', 'tag', 'meth', 'FromIdx', 'ToIdx'],
        "Telemetry_ModelPair.csv": ['TrainModel','TrainData','TrainInvo','TrainLine','TrainMethod','TrainCtx',
                                    'TestModel','TestData','TestInvo','TestLine','TestMethod','TestCtx'],
        "InvokeInjected.csv":['Invocation','Method','InMeth'],
        "FLowVarStoreIndex.csv": ['To', 'ToCtx', 'Instr', 'From','FromCtx', 'tag', 'meth', 'FromIdx', 'ToIdx'],
    }
    
    return d[filename]

def read_csv_or_empty(fact_path,filename):

    filepath= os.path.join(fact_path,filename)
    if os.path.getsize(filepath) > 0:
        return pd.read_csv(filepath, sep="\t", names=get_columns(filename))
    else:
        return pd.DataFrame()



def read_index_or_empty(df,index):

    if df.empty:
        return []
    else:
        return list(df[index])
    




def match_invo(label,injected_invos):

    numbers = re.findall("\d+", label)
    for number in numbers:
        shift = sum(1 for el in injected_invos if el < int(number))
        label= re.sub(number,str(int(number)-shift),label)

    return label



if __name__=='__main__':


    fact_path= sys.argv[1]
    file_path= sys.argv[2]
    
    FlowVarTransformation= read_csv_or_empty(fact_path,"FLowVarTransformation.csv")
    FlowVarStoreIndex= read_csv_or_empty(fact_path,"FLowVarStoreIndex.csv")
    df_injected= read_csv_or_empty(fact_path,"InvokeInjected.csv")
    df_telemetry_model_pair= read_csv_or_empty(fact_path,"Telemetry_ModelPair.csv")


    if(FlowVarTransformation.empty==False):
        FlowVarTransformation["To"]=FlowVarTransformation["To"].fillna("")
        FlowVarTransformation["From"]=FlowVarTransformation["From"].fillna("")
        FlowVarTransformation["InstructionId"] = FlowVarTransformation["ToCtx"] + FlowVarTransformation["Instr"].astype(str) + FlowVarTransformation["FromCtx"]
        FlowVarTransformation["ToId"] = FlowVarTransformation["To"] + FlowVarTransformation["ToCtx"]
        FlowVarTransformation["FromId"] = FlowVarTransformation["From"] + FlowVarTransformation["FromCtx"]

    if(FlowVarStoreIndex.empty==False):
        FlowVarStoreIndex["To"]=FlowVarStoreIndex["To"].fillna("")
        FlowVarStoreIndex["From"]=FlowVarStoreIndex["From"].fillna("")
        FlowVarStoreIndex["InstructionId"] = FlowVarStoreIndex["ToCtx"] + FlowVarStoreIndex["Instr"].astype(str) + FlowVarStoreIndex["FromCtx"]
        FlowVarStoreIndex["ToId"] = FlowVarStoreIndex["To"] + FlowVarStoreIndex["ToCtx"]
        FlowVarStoreIndex["FromId"] = FlowVarStoreIndex["From"] + FlowVarStoreIndex["FromCtx"]


    if(df_telemetry_model_pair.empty==False):
        df_telemetry_model_pair['TrainInstr']=df_telemetry_model_pair['TrainCtx']+df_telemetry_model_pair['TrainLine'].astype(str)+df_telemetry_model_pair['TrainCtx']
        df_telemetry_model_pair['TestInstr']=df_telemetry_model_pair['TestCtx']+df_telemetry_model_pair['TestLine'].astype(str)+df_telemetry_model_pair['TestCtx']
        df_telemetry_model_pair['TrainVar']=df_telemetry_model_pair['TrainData']+df_telemetry_model_pair['TrainCtx']
        df_telemetry_model_pair['TestVar']=df_telemetry_model_pair['TestData']+df_telemetry_model_pair['TestCtx']


    


    unique_instr,unique_vars,index_mapping = extract_instr_and_vars(FlowVarTransformation,FlowVarStoreIndex)
    flow_from_inst={ instr:[] for instr in unique_instr}
    flow_to_inst={instr:[] for instr in unique_instr}
    instr_labels={}
    var_labels={}

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise FileNotFoundError

    except Exception as e:
        print(f"An error occurred: {e}")
        raise Exception

    instr_pattern = r'(?<=\])\d+'
    var_pattern = r'.*(?=\[)'


    for index,row in FlowVarTransformation.iterrows():
        if(row["InstructionId"] not in instr_labels.keys()):
                instr_labels[row["InstructionId"]]=' '.join(row["tag"].split()[:2])
        if re.fullmatch("\[(\$?invo\d+?)?, (\$?invo\d+?)?\]",row["ToId"])==None:
            flow_from_inst[row["InstructionId"]].append(row["ToId"])
        if re.fullmatch("\[(\$?invo\d+?)?, (\$?invo\d+?)?\]",row["FromId"])==None:
            flow_to_inst[row["InstructionId"]].append(row["FromId"])
            if(row["FromId"] not in var_labels.keys()):
                var_labels[row["FromId"]]=' '.join(row["tag"].split()[2:])


    for index,row in FlowVarStoreIndex.iterrows():
        if(row["InstructionId"] not in instr_labels.keys()):
                instr_labels[row["InstructionId"]]=' '.join(row["tag"].split()[:2])
        if re.fullmatch("\[(\$?invo\d+?)?, (\$?invo\d+?)?\]",row["ToId"])==None:
            flow_from_inst[row["InstructionId"]].append(row["ToId"])
        if re.fullmatch("\[(\$?invo\d+?)?, (\$?invo\d+?)?\]",row["FromId"])==None:
            flow_to_inst[row["InstructionId"]].append(row["FromId"])
            if(row["FromId"] not in var_labels.keys()):
                var_labels[row["FromId"]]=' '.join(row["tag"].split()[2:])



    flow_from_inst = {instr:list(set(vars)) for instr,vars in flow_from_inst.items()}
    flow_to_inst = {instr:list(set(vars)) for instr,vars in flow_to_inst.items()}


    
    adj=build_adj(flow_from_inst,flow_to_inst, unique_instr,unique_vars)

    del flow_from_inst, flow_to_inst

    if not os.path.exists(os.path.join(fact_path,'_graphs')):
        os.makedirs(os.path.join(fact_path,'_graphs'))
        print("created",fact_path+'/_graphs')
    else:
        remove_files(os.path.join(fact_path,'_graphs'))
    graph_paths= os.path.join(fact_path,"_graphs")


    G = dgl.graph(np.where(adj>0))



    # Create an empty DataFrame
    labels = pd.DataFrame(index=range(len(index_mapping)),columns=['Nodes', 'Labels'])
    for key, value in instr_labels.items():
        labels.iloc[index_mapping[key]] = {'Nodes': key, 'Labels': value}
    for key, value in var_labels.items():
        labels.iloc[index_mapping[key]] = {'Nodes': key, 'Labels': value}
    labels.fillna("",inplace=True)

    del instr_labels, var_labels

    if(df_telemetry_model_pair.empty==False):
        injected_invos = list(map(lambda x : int(re.search("\d+", x).group()), df_injected["Invocation"])) if df_injected.empty == False else []

        for index,row in df_telemetry_model_pair.iterrows():
            original=row['TrainInvo']+"_"+row['TestInvo']+"_"+row['TrainCtx']+"_"+row['TestCtx']
            pair = [row['TrainInstr'],row['TestInstr']]
            binary_labels= labels.copy()
            try:
                binary_labels.iloc[index_mapping[row['TrainInstr']],1]+= ' Train'
                binary_labels.iloc[index_mapping[row['TestInstr']],1]+= ' Test'
                binary_labels.iloc[index_mapping[row['TrainVar']],1]+= ' TrainData'
                binary_labels.iloc[index_mapping[row['TestVar']],1]+= ' TestData'
            except KeyError:
                print(f'The model pair [{row["TrainModel"]},{row["TestModel"]}], skipping it...')
                continue

            features = create_binary_features(binary_labels)

            G.ndata['features']= features
            original = match_invo(original,injected_invos) if len(injected_invos)>0 else original


            pair_node_id = list(map(lambda x: index_mapping[x],pair))
            sg,_ = dgl.khop_in_subgraph(G, pair_node_id,k=G.num_edges(),relabel_nodes=True)
            save_graphs(os.path.join(graph_paths,f'{original}.bin'), [sg])



        



    