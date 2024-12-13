import os
import json
#import uuid

if __name__ == "__main__":
    AFO_RESOURCE_CONFIG = os.environ.get('AFO_RESOURCE_CONFIG')
    AFO_ENV_CLUSTER_SPEC = os.environ.get('AFO_ENV_CLUSTER_SPEC')
    
    resource_json_obj = json.loads(AFO_RESOURCE_CONFIG)
    gpu_count = resource_json_obj['worker']['gpu']
    cluster_spec_json_obj = json.loads(AFO_ENV_CLUSTER_SPEC)
    worker_list = cluster_spec_json_obj['worker']
    
    #fileuuid = uuid.uuid4()
    #hostfile_path = os.path.join('/workdir', str(fileuuid))
    hostfile_path = os.path.join('/workdir', 'TMPHOSTFILE')
    with open(hostfile_path, 'w') as f:
        for worker in worker_list:
            f.write(worker.split(':')[0]+' slots='+str(gpu_count)+'\n')
    print('hostfile path is '+hostfile_path)
