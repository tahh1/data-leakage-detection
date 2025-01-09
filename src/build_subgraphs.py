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
from .utils import remove_files

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

def create_code_embeddings(df,tokenizer,sent2vec_model):
     features = list(map(lambda code: code.strip(),df.iloc[:,2].to_list()))
     #print(features)
     lines_content_tokenized = [tokenizer.encode(line).tokens for line in features]
     #print(lines_content_tokenized)
     code_embeddings = [sent2vec_model.embed_sentence(" ".join(line_content_tokenized))[0] for line_content_tokenized in lines_content_tokenized]
     return torch.from_numpy(np.array(code_embeddings)).to(torch.float)


def build_adj(flow_from_inst,flow_to_inst, unique_instr,unique_vars,index_mapping):

    adj= np.zeros((len(unique_instr)+len(unique_vars),len(unique_instr)+len(unique_vars)))
    for index,instr in enumerate(unique_instr):
        for to_vars in flow_from_inst[instr]:
            adj[index][index_mapping[to_vars]]=1
        for from_vars in flow_to_inst[instr]:
            adj[index_mapping[from_vars]][index]=1

    return adj


def build_adj_instr_only(flow_from_inst,flow_to_inst, unique_instr,unique_vars,index_mapping):

    adj= np.zeros((len(unique_instr),len(unique_instr)))
    for index,instr in enumerate(unique_instr):
        for to_vars in flow_from_inst[instr]:
            for to_instr in flow_to_inst.keys():
                if(to_vars in flow_to_inst[to_instr]):
                    adj[index][index_mapping[to_instr]]=1
        # for from_vars in flow_to_inst[instr]:
        #     adj[index_mapping[from_vars]][index]=1

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



def build_subgraphs(fact_path,file_path,tokenizer,sent2vec_model):


    #fact_path= sys.argv[1]
    #file_path= sys.argv[2]
    
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
    instr_loc={}
    var_loc={}
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
            #print(row["InstructionId"],re.findall(instr_pattern, row["InstructionId"])[0],lines[int(re.findall(instr_pattern, row["InstructionId"])[0])-1])
            instr_loc[row["InstructionId"]] = lines[int(re.findall(instr_pattern, row["InstructionId"])[0])-1]
        if re.fullmatch("\[(\$?invo\d+?)?, (\$?invo\d+?)?\]",row["ToId"])==None:
            flow_from_inst[row["InstructionId"]].append(row["ToId"])
            var_loc[row["ToId"]] = lines[int(re.findall(instr_pattern, row["InstructionId"])[0])-1]
        if re.fullmatch("\[(\$?invo\d+?)?, (\$?invo\d+?)?\]",row["FromId"])==None:
            flow_to_inst[row["InstructionId"]].append(row["FromId"])
            if(row["FromId"] not in var_labels.keys()):
                var_labels[row["FromId"]]=' '.join(row["tag"].split()[2:])


    for index,row in FlowVarStoreIndex.iterrows():
        if(row["InstructionId"] not in instr_labels.keys()):
                instr_labels[row["InstructionId"]]=' '.join(row["tag"].split()[:2])
                instr_loc[row["InstructionId"]] = lines[int(re.findall(instr_pattern, row["InstructionId"])[0])-1]
        if re.fullmatch("\[(\$?invo\d+?)?, (\$?invo\d+?)?\]",row["ToId"])==None:
            flow_from_inst[row["InstructionId"]].append(row["ToId"])
            var_loc[row["ToId"]] = lines[int(re.findall(instr_pattern, row["InstructionId"])[0])-1]
        if re.fullmatch("\[(\$?invo\d+?)?, (\$?invo\d+?)?\]",row["FromId"])==None:
            flow_to_inst[row["InstructionId"]].append(row["FromId"])
            if(row["FromId"] not in var_labels.keys()):
                var_labels[row["FromId"]]=' '.join(row["tag"].split()[2:])



    flow_from_inst = {instr:list(set(vars)) for instr,vars in flow_from_inst.items()}
    flow_to_inst = {instr:list(set(vars)) for instr,vars in flow_to_inst.items()}


    
    adj=build_adj(flow_from_inst,flow_to_inst, unique_instr,unique_vars,index_mapping)
    adj_instr=build_adj_instr_only(flow_from_inst,flow_to_inst, unique_instr,unique_vars,index_mapping)
    del flow_from_inst, flow_to_inst

    if not os.path.exists(os.path.join(fact_path,'_graphs')):
        os.makedirs(os.path.join(fact_path,'_graphs'))
        print("created",fact_path+'/_graphs')
    else:
        remove_files(os.path.join(fact_path,'_graphs'))
    graph_paths= os.path.join(fact_path,"_graphs")
    Adj_df = pd.DataFrame(adj, index=unique_instr+unique_vars, columns=unique_instr+unique_vars)
    Adj_df_instr = pd.DataFrame(adj_instr, index=unique_instr, columns=unique_instr)
    print(len(Adj_df_instr))
    np.transpose(Adj_df).to_csv(os.path.join(graph_paths,"graph_unpruned.csv"))
    np.transpose(Adj_df_instr).to_csv(os.path.join(graph_paths,"graph_unpruned_instr_only.csv"))


    G = dgl.graph(np.where(adj>0), num_nodes=len(adj))
    G_instr = dgl.graph(np.where(Adj_df_instr>0), num_nodes=len(Adj_df_instr))
    print(G_instr.num_nodes())



    # Create an empty DataFrame
    labels = pd.DataFrame(index=range(len(index_mapping)),columns=['Nodes', 'Labels', 'Code'])
    for key, value in instr_labels.items():
        labels.iloc[index_mapping[key]] = {'Nodes': key, 'Labels': value, 'Code': instr_loc[key] if key in instr_loc.keys() else ""}
    for key, value in var_labels.items():
        labels.iloc[index_mapping[key]] = {'Nodes': key, 'Labels': value, 'Code': var_loc[key] if key in var_loc.keys() else ""}

    labels.fillna("",inplace=True)

    labels.to_csv(f"{graph_paths}/features_unpruned.csv")
    labels.iloc[:len(unique_instr)].to_csv(f"{graph_paths}/features_instr_unpruned.csv")

    del instr_labels, var_labels

    if(df_telemetry_model_pair.empty==False):
        injected_invos = list(map(lambda x : int(re.search("\d+", x).group()), df_injected["Invocation"])) if df_injected.empty == False else []
        print("embedding code...")
        code_embedding = create_code_embeddings(labels,tokenizer,sent2vec_model)
        print(code_embedding.shape)
        code_embedding_instr = code_embedding[:len(unique_instr)]

        for index,row in df_telemetry_model_pair.iterrows():
            original=row['TrainInvo']+"_"+row['TestInvo']+"_"+row['TrainCtx']+"_"+row['TestCtx']
            pair = [row['TrainInstr'],row['TestInstr']]
            feature_labels= labels.copy()
            try:
                feature_labels.iloc[index_mapping[row['TrainInstr']],1]+= ' Train'
                feature_labels.iloc[index_mapping[row['TestInstr']],1]+= ' Test'
                feature_labels.iloc[index_mapping[row['TrainVar']],1]+= ' TrainData'
                feature_labels.iloc[index_mapping[row['TestVar']],1]+= ' TestData'
            except KeyError:
                print(f'The model pair [{row["TrainModel"]},{row["TestModel"]}], skipping it...')
                continue
            
            binary_features = create_binary_features(feature_labels)
            print(binary_features.shape,code_embedding.shape)
            binary_features_instr = binary_features[:len(unique_instr)]
            

            #G.ndata['features']= binary_features
            #G_instr.ndata['features']= binary_features_instr
            original = match_invo(original,injected_invos) if len(injected_invos)>0 else original


            pair_node_id = list(map(lambda x: index_mapping[x],pair))

            # sg,_ = dgl.khop_in_subgraph(G, pair_node_id,k=G.num_edges(),relabel_nodes=True)
            sg_instr,_ = dgl.khop_in_subgraph(G_instr, pair_node_id,k=G.num_edges(),relabel_nodes=True)



            # adj_sg = dgl.khop_adj(sg,1)
            # _ID = sg.ndata["_ID"]
            # kept_nodes = [node for index, node in enumerate(unique_instr + unique_vars) if index in _ID]
            # A = pd.DataFrame(adj_sg, index=kept_nodes, columns=kept_nodes)
            # np.transpose(A).to_csv(os.path.join(graph_paths,f"{original}.csv"))
            # feature_labels.iloc[_ID].to_csv(f"{graph_paths}/{original}_features.csv")

            #sg.ndata["features"]= torch.cat((binary_features[_ID],code_embedding[_ID]),dim = 1)
            #print(sg,sg.ndata["features"].shape)
            #save_graphs(os.path.join(graph_paths,f'{original}_original_both.bin'), [sg])
            #sg.ndata["features"]= binary_features[_ID]
            # print(sg)
            # save_graphs(os.path.join(graph_paths,f'{original}_original_binary.bin'), [sg])
            # sg.ndata["features"]= code_embedding[_ID]
            # print(sg)
            # save_graphs(os.path.join(graph_paths,f'{original}_original_embeddings.bin'), [sg])

            adj_sg = dgl.khop_adj(sg_instr,1)
            _ID = sg_instr.ndata["_ID"]
            kept_nodes = [node for index, node in enumerate(unique_instr + unique_vars) if index in _ID]
            A = pd.DataFrame(adj_sg, index=kept_nodes, columns=kept_nodes)
            np.transpose(A).to_csv(os.path.join(graph_paths,f"{original}_inst.csv"))
            feature_labels.iloc[_ID].to_csv(f"{graph_paths}/{original}_inst_features.csv")

            sg_instr.ndata["features"]= torch.cat((binary_features_instr[_ID],code_embedding_instr[_ID]),dim=1)
            print(sg_instr)
            save_graphs(os.path.join(graph_paths,f'{original}_instr_both.bin'), [sg_instr])
            sg_instr.ndata["features"]= binary_features_instr[_ID]
            print(sg_instr)
            save_graphs(os.path.join(graph_paths,f'{original}_instr_binary.bin'), [sg_instr])
            sg_instr.ndata["features"]= code_embedding_instr[_ID]
            print(sg_instr)
            save_graphs(os.path.join(graph_paths,f'{original}_instr_embeddings.bin'), [sg_instr])
            
            



        



    
