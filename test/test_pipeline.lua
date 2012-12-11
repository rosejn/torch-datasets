require 'fn/seq'
require 'dataset'
require 'dataset/table_dataset'

function tests.test_sampler()
    local dset = {data = torch.Tensor({{1}, {2}, {3}, {4}, {5}})}
    local td = dataset.TableDataset(dset)
    local sampler = td:sampler()
    local samples = seq.table(seq.map(function(s) return s.data[1] end, seq.take(5, sampler)))
    table.sort(samples)
    tester:assertTableEq(samples, {1,2,3,4,5}, "sample a dataset")
end

