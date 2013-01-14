require 'torch'
require 'unsup'
require 'dataset'


-- ZCA-Whitening 
function dataset.zca_whiten(sample)

	local display = false

	local dims = sample.data:size()
	local nsamples = dims[1]
	local n_dimensions = dims[2] * dims[3] * dims[4]
	
	local mdata = sample.data:clone():reshape(nsamples, n_dimensions)
	mdata:add(torch.ger(torch.ones(nsamples), torch.mean(mdata, 1):squeeze()):mul(-1))
	local ce, cv = unsup.pcacov(sample.data:reshape(nsamples, n_dimensions))
	local val = ce:clone():add(1e-1):sqrt():pow(-1)
	local diag = torch.diag(val)
	local P = torch.mm(cv, diag)
	P = torch.mm(P, cv:t())
	local wdata = torch.mm(mdata, P)
	local mwdata = wdata:clone():add(torch.ger(torch.ones(nsamples), torch.mean(wdata, 1):squeeze()):mul(-1))
	sample.data:copy(wdata:reshape(dims[1], dims[2], dims[3], dims[4]):typeAs(sample.data))
	return torch.mean(mdata, 1), P
end

