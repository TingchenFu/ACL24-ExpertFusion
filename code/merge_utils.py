import torch
import copy
from collections import OrderedDict


def statedict2vector(state_dict, remove_keys=[], keep_keys=[]):
    # remove keys or keep keys from the state_dict, and then transfer to a torch vector
    shared_state_dict = state_dict
    #shared_state_dict = copy.deepcopy(state_dict)
    if len(remove_keys):
        for key in remove_keys:
            if key in shared_state_dict:
                del shared_state_dict[key]
    elif len(keep_keys):
        for key in list(shared_state_dict.keys()):
            if key not in keep_keys:
                del shared_state_dict[key]
    else:
        # only no change in state dict
        pass
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    #print(list(sorted_shared_state_dict.keys())[:100])
    #print(list(sorted_shared_state_dict['roberta.encoder.layer.0.attention.self.key.bias']))
    
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )


def vector2statedict(vector, state_dict, remove_keys=[], keep_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)



    if len(remove_keys):
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
    elif len(keep_keys):
        for key in list(reference_dict.keys()):
            if key not in keep_keys:
                del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))
    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        sorted_reference_dict["transformer.encoder.embed_tokens.weight"] = sorted_reference_dict["transformer.shared.weight"]
        sorted_reference_dict["transformer.decoder.embed_tokens.weight"] = sorted_reference_dict["transformer.shared.weight"]
    return sorted_reference_dict



def aggregate(T, agg_type, ratio=None, final_signs=None, dim=0):
    if isinstance(T,list):
        T = torch.stack(T,dim=0)
    #print(T.shape)
    if agg_type == "mean":
        result = torch.mean(T, dim=dim)
    elif agg_type == "sum":
        result = torch.sum(T, dim=dim)
    elif agg_type == "median":
        result = torch.median(T, dim=dim)[0]
    elif agg_type == "magnitude":
        max_indices = T.abs().argmax(dim=0)
        result = T[max_indices, torch.arange(T.shape[1])]
    elif agg_type=='weighted':
        assert ratio is not None
        ratio = torch.tensor(ratio,device=T.device).unsqueeze(1)
        result = (T*ratio).sum(dim=dim)
    else:
        raise ValueError("Invalid agg_type: %s" % agg_type)

    if final_signs is not None:
        # print(final_signs)
        result = result.abs() * final_signs
    return result



def basic_merging(agg_mode, ratio, finetune_statedicts, remove_keys):
    # ft : the finetuned checkpoint state dict
    """ 
    "Basic aggregation of the delta checks
    implemented by Prateek Yadav
    """
    # probably not necessary
    #all_checks = flat_checks.clone()
    finetune_vectors=[]
    for state_dict in finetune_statedicts:
        finetune_vector = statedict2vector(state_dict,remove_keys).to('cpu')
        print(finetune_vector.device)
        finetune_vectors.append(finetune_vector)
        del state_dict
        del finetune_vector
        torch.cuda.empty_cache()
        
    #(n_model,n_theta)
    finetune_vectors = torch.stack(finetune_vectors)
    merged_vector = aggregate(finetune_vectors, agg_mode, ratio, final_signs=None)
    merged_statedict = vector2statedict(merged_vector, finetune_statedicts[0], remove_keys=remove_keys)
    return merged_statedict