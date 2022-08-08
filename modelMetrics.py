# basic imports
import os
import sys
import time
import numpy as np

# DL library imports
import torch
import torch.nn as nn

# import profiling package
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/thop_library')
from thop import profile
from thop import clever_format


## constants
KILOBYTE_TO_BYTE = 1024
MEGABYTE_TO_KILOBYTE = 1024


class inferenceMetrics:
    def __init__(self) -> None:
        self.MACs = 'NA'
        self.FLOPs = 'NA'
        self.params = 'NA'
        self.fileSize = 'NA'
        self.modelSize = 'NA'
        self.computeDevice = 'NA'
        self.inferenceTime = 'NA'

    def __repr__(self) -> str:
        computeString = f"Device = {self.computeDevice}"

        mmacs = 'NA'
        if self.MACs != 'NA':
            mmacs = f'{self.MACs/1e6}'
        mflops = 'NA'
        if self.FLOPs != 'NA':
            mflops = f'{self.FLOPs/1e6}'
        mparams = 'NA'
        if self.params != 'NA':
            mparams = f'{self.params/1e6}'

        paramString = f"{mmacs} MMACs, {mflops} MFLOPs and {mparams} M parameters"
        # sizeString = f"Memory size(params + buffers) = {self.modelSize} MB, FileSize = {self.fileSize} MB"
        sizeString = f"Model FileSize = {self.fileSize} MB"

        time = 'NA'
        if self.inferenceTime != 'NA':
            time = self.inferenceTime * 1e3
        timeString = f"Single batch inference Time of model = {time} milliseconds"
        return computeString + "\n" +  paramString + "\n" + sizeString + "\n" + timeString


def findNumOfTrainableParams(model : nn.Module):
    """ Function returns the number of weights (trainable parameters) in a model;
    A parameter is considered a weight if its gradient needs to be calculated during BackProp.

    Args:
        model (nn.Module): input model

    Returns:
        totalNumTrainableParams : number of parameters in model for which gradient is to be
        computed during backprop
    """    
    totalNumTrainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return totalNumTrainableParams


def getModelFileSize(model : nn.Module) -> float:
    """function returns size of model state dict in MB 
    Args:
        model (nn.Module): input model
    Returns:
        modelFileSize (float): size of model state dict in MB
    """    

    # check if its input module is eager / script mode
    scriptVersion = isinstance(model, (torch.jit.ScriptModule, torch.jit.ScriptFunction))

    # save model to file and find the size
    if scriptVersion == True:
        torch.jit.save(torch.jit.script(model), "temp.p")
    else:
        torch.save(model.state_dict(), "temp.p")

    modelFileSize = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return modelFileSize


def computeInferenceTime(model : nn.Module, input : torch.Tensor, device:torch.device, nIters = 100) -> float:
    """Function to calculate inference time of model on given input

    Args:
        model (nn.Module): input model
        input (torch.Tensor): sample input of valid shape
        device (torch.device): compute device as in CPU, GPU, TPU]. Defaults to 'cpu'.
        nIters (int, optional): number of iterations over which to find avg inference time. Defaults to 100.

    Returns:
        avgInferenceTime (float): Avg inference time for `input` over `nIters` for `model`
    """   
    # initialize default value
    avgInferenceTime = np.inf

    # check for GPU availability and user option
    checkForGPU = False
    if torch.cuda.is_available() == True:
        if 'cuda' in str(device):
            checkForGPU = True

    # check for zero input
    if nIters > 0:
        # move to target device        
        model = model.to(device)
        input = input.to(device)

        # change model to inference mode
        model.eval()

        # find the avg time take for forward pass
        with torch.no_grad():
            start_time = time.time()
            for _ in range(nIters):
                _ = model(input)

                # wait for cuda to finish (cuda is asynchronous!)
                if checkForGPU == True:
                    torch.cuda.synchronize()
            endTime = time.time()
        
        elapsedTime = endTime - start_time
        batchSize = input.size()[0]
        avgInferenceTime = elapsedTime / (batchSize *  nIters)
    return avgInferenceTime 



def getModelSize(model : nn.Module) -> float:
    """ Function calculates size occupied by model parameters and buffers in MB

    Args:
        model (nn.Module) : module to find size of

    Returns:
        modelSize (float) : model size in MB
    """    
    paramsMemory = sum([param.nelement()*param.element_size() for param in model.parameters()])
    bufferMemory = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    modelSize = paramsMemory + bufferMemory # in bytes
    modelSize = modelSize / (KILOBYTE_TO_BYTE * MEGABYTE_TO_KILOBYTE)
    return modelSize

    

def modelMetrics(model : nn.Module, modelName :str, input : torch.Tensor, 
                 device : torch.device, nIters : int =10, verbose :bool =False) -> inferenceMetrics:
    """ Function calculates following metrics of model
        1. MACs - number of Multiply-accumulate computations
        2. FLOPs - number of Floating point operations
        3. params - number of parameters in the model
        4. Model Size - memory occupied by Model parameters and buffers 
        5. inference time - Avg Time taken for single batch inference
        6. File Size - disk size of the model 

    Args:
        model (nn.Module): input module to compute metrics for
        modelName (str): name of input model
        input (torch.Tensor): sample input of valid shape for finding inference time
        device (torch.device): indicates whether CPU or GPU to use
        nIters (int) : number of iterations to calculate avg inference time on
        verbose (bool) : flag to print output
            
    Returns:
        metrics (inferenceMetrics) : class containing info w.r.t model inference
    """
    print(f"-------------\n Metrics of {modelName} \n-------------")

    model.eval()    
    model.to(device)
    input.to(device)
    metrics = inferenceMetrics()
    try:
        metrics.computeDevice = str(device)
    except:
        pass

    try:
        metrics.MACs, metrics.params = profile(model, inputs=(input,), verbose=False)
        metrics.FLOPs = 2 * metrics.MACs
    except:
        pass

    try:
        metrics.modelSize = getModelSize(model)
    except:
        pass

    try:
        metrics.inferenceTime = computeInferenceTime(model, input, device, nIters=nIters)
    except:
        pass

    try:
        metrics.fileSize = getModelFileSize(model)
    except:
        pass

    if verbose == True:
        print(metrics)
    return metrics