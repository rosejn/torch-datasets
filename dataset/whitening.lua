require 'torch'
require 'fs'
require 'xlua'
require 'logroll'
require 'dataset'
require 'dataset/pipeline'
require 'dataset/lushio'
require 'dataset/table_dataset'

local function printImageStats(data, _name, noGUI, _wx, _wy)
	local name = _name or ''
	local wx = _wx or 32
	local wy = _wy or 32

	-- print stats
	print(' ')
	print(name)
	print{data}
	print('Stats:')
	print('Min('..name..'): ', torch.min(data))
	print('Mean('..name..'): ', torch.mean(data))
	print('Max('..name..'): ', torch.max(data))
	print('Std('..name..'): ', torch.std(data))
	print(' ')
	
	if noGUI then
		local data = data:unfold(2, wx, wy)
		--image.display({image=data[{{1, 144}, {}, {}}], nrow = 16,  symmetric=true, min=-5, max=5, zoom=1, legend=name, padding=2})
		image.display({image=data[{{1, 144}, {}, {}}], nrow = 16,  symmetric=false, zoom=3, legend=name, padding=2})
		--image.display({image=data[{{1, 144}, {}, {}}], nrow = 16, zoom=3, legend=name, padding=2})
		--image.display({image=data[{{1, 3600}, {}, {}}], nrow = 16*5, symmetric=true, min=-5, max=5, legend=name, padding=2})
		--image.display({image=data[{{1, 3600}, {}, {}}], nrow = 16*5, legend=name, padding=2})
	end
end


local function printMatStats(data, _name, noGUI)
	local name = _name or ''

	-- print stats
	print('')
	print(name)
	print{data}
	print('Stats:')
	print('Min('..name..'): ', torch.min(data))
	print('Mean('..name..'): ', torch.mean(data))
	print('Max('..name..'): ', torch.max(data))
	print('Std('..name..'): ', torch.std(data))
	print(' ')
	
	if noGUI then
		image.display({image=data, symmetric=true, legend=name})
	end

end



-- ZCA-Whitening 
function  dataset.zca_whiten(sample)

	local display = false

	local dims = sample.data:size()
	local nsamples = dims[1]
	local n_dimensions = dims[2] * dims[3] * dims[4]
	
	require 'unsup'
	-- compute the covariance matrix by hand
	local mdata = sample.data:clone():reshape(nsamples, n_dimensions)
	--printImageStats(mdata, 'mdata', display, dims[3], dims[4])
	mdata:add(torch.ger(torch.ones(nsamples), torch.mean(mdata, 1):squeeze()):mul(-1))
	--local covdata = torch.mm(mdata:t(), mdata)
	--printMatStats(covdata, 'covdata', display)
	local ce, cv = unsup.pcacov(sample.data:reshape(nsamples, n_dimensions))
	local val = ce:clone():add(1e-1):sqrt():pow(-1)
	local diag = torch.diag(val)
	local P = torch.mm(cv, diag)
	P = torch.mm(P, cv:t())
	local wdata = torch.mm(mdata, P)
	--printImageStats(wdata, 'wdata', display, dims[3], dims[4])
	local mwdata = wdata:clone():add(torch.ger(torch.ones(nsamples), torch.mean(wdata, 1):squeeze()):mul(-1))
	--local covwdata = torch.mm(mwdata:t(), mwdata)
	--printMatStats(covwdata, 'covwdata', display)
	--print(dims)
	--print{wdata}
	sample.data:copy(wdata:reshape(dims[1], dims[2], dims[3], dims[4]):typeAs(sample.data))
	return torch.mean(mdata, 1), P
end

