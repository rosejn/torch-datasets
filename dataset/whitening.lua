require 'torch'
require 'unsup'
require 'dataset'


-- ZCA-Whitening
function dataset.zca_whiten(data, _P)

    local P = _P
    local invP

    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions = dims[2] * dims[3] * dims[4]

    local mdata = data:clone():reshape(nsamples, n_dimensions)
    mdata:add(torch.ger(torch.ones(nsamples), torch.mean(mdata, 1):squeeze()):mul(-1))
    if not P then
        local ce, cv = unsup.pcacov(data:reshape(nsamples, n_dimensions))
        local invval = ce:clone():add(1e-1):sqrt()
        local val = invval:clone():pow(-1)
        local diag = torch.diag(val)
        P = torch.mm(cv, diag)
        P = torch.mm(P, cv:t())

        local invdiag = torch.diag(invval)
        invP = torch.mm(cv:t(), invdiag)
        invP = torch.mm(invP, cv)
    end
    local wdata = torch.mm(mdata, P)
    return wdata:reshape(dims[1], dims[2], dims[3], dims[4]):typeAs(data), P, invP
end

