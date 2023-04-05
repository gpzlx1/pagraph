import torch

torch.ops.load_library("./build/libpg.so")

cache_nids = torch.tensor([5, 6, 89, 34, 2]).int().cuda()

for i in torch.ops.pg_ops._CAPI_create_hashmap(cache_nids):
    print(i)

cpu_data = torch.arange(0, 10000).reshape(100, 100).float()
cache_nids = torch.randint(0, 100, (20, )).int().cuda()
gpu_data = cpu_data[cache_nids.cpu().long()].cuda()

hashmap_tensor, hashmap_value = torch.ops.pg_ops._CAPI_create_hashmap(
    cache_nids)

index = torch.randint(0, 100, (100, )).long().cuda()
torch.ops.pg_ops._CAPI_pin_tensor(cpu_data)

print(cpu_data[index.long()].cuda().equal(
    torch.ops.pg_ops._CAPI_fetch_feature_data_with_caching(
        cpu_data, gpu_data, hashmap_tensor, hashmap_value, index)))
