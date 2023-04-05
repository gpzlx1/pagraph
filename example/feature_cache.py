import torch
import time

torch.ops.load_library("./build/libpg.so")


class FeatureCacheServer:

    def __init__(self, feature):
        self.feature = feature
        self.cached_feature = None
        self.hashmap_key = None
        self.hashmap_value = None
        self.full_cached = False
        self.no_cached = False

    def cache_feature(self, cache_nids, full_cached=False):
        self.full_cached = full_cached

        start = time.time()
        cache_size = 0

        if self.full_cached:
            self.cached_feature = self.feature.cuda()
            cache_size = self.cached_feature.numel(
            ) * self.cached_feature.element_size()

        else:
            if cache_nids.numel() > 0:
                self.cached_feature = self.feature[
                    cache_nids.cpu().long()].cuda()
                self.hashmap_key, self.hashmap_value = torch.ops.pg_ops._CAPI_create_hashmap(
                    cache_nids.cuda())
                cache_size = self.cached_feature.numel(
                ) * self.cached_feature.element_size()

            else:
                self.no_cached = True
                cache_size = 0

            torch.ops.pg_ops._CAPI_pin_tensor(self.feature)

        torch.cuda.synchronize()
        end = time.time()

        print(
            "GPU {} takes {:.3f} s to cache feature data, cached size = {:.3f} GB, cache rate = {:.3f}"
            .format(
                torch.cuda.current_device(), end - start,
                cache_size / 1024 / 1024 / 1024, cache_size /
                (self.feature.element_size() * self.feature.numel())))

    def fetch_data(self, index):
        if self.full_cached:
            return torch.index_select(self.cached_feature, 0, index.cuda())
        elif self.no_cached:
            return torch.ops.pg_ops._CAPI_fetch_feature_data(
                self.feature, index)
        else:
            return torch.ops.pg_ops._CAPI_fetch_feature_data_with_caching(
                self.feature,
                self.cached_feature,
                self.hashmap_key,
                self.hashmap_value,
                index.cuda(),
            )

    def __del__(self):
        torch.ops.pg_ops._CAPI_unpin_tensor(self.feature)


if __name__ == "__main__":
    cpu_data = torch.arange(0, 10000).reshape(100, 100).float()

    full_cache_server = FeatureCacheServer(cpu_data)
    full_cache_server.cache_feature(torch.tensor([]).int().cuda(), True)

    no_cache_server = FeatureCacheServer(cpu_data)
    no_cache_server.cache_feature(torch.tensor([]).int().cuda(), False)

    cache_nids = torch.randint(0, 100, (20, )).int().cuda()
    part_cache_server = FeatureCacheServer(cpu_data)
    part_cache_server.cache_feature(cache_nids, False)

    index = torch.randint(0, 100, (100, )).long().cuda()
    print(full_cache_server.fetch_data(index))
    #print(no_cache_server.fetch_data(index))
    print(part_cache_server.fetch_data(index))