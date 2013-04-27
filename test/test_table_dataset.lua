require 'fn/seq'
require 'dataset'
require 'dataset/TableDataset'

function tests.test_sampler()
    local dset = {data = torch.Tensor({{1}, {2}, {3}, {4}, {5}})}
    local td = dataset.TableDataset(dset)
    local sampler = td:sampler()
    local samples = seq.table(seq.map(function(s) return s.data[1] end, seq.take(5, sampler)))
    table.sort(samples)
    tester:assertTableEq(samples, {1,2,3,4,5}, "sample a dataset")
end

function tests.test_binarize()
    local dset = {data = torch.Tensor({{1}, {2}, {3}, {4}, {5}})}
    local td = dataset.TableDataset(dset)
    local sampler = td:sampler()
    local samples = seq.table(seq.map(function(s) return s.data[1] end, seq.take(5, sampler)))
    table.sort(samples)
    tester:assertTableEq(samples, {1,2,3,4,5}, "dataset wrong before binarization")

    td:binarize(3)
    sampler = td:sampler()
    samples = seq.table(seq.map(function(s) return s.data[1] end, seq.take(5, sampler)))
    table.sort(samples)
    tester:assertTableEq(samples, {0,0,1,1,1}, "dataset wrong after binarization")
end
