lushio = {}

function lushio.read(filename)
   -- Reads Lush binary formatted matrix and returns it.
   -- The matrix is stored in 'filename'.
   --
   --   call : x = luahio.readBinaryLushMatrix('my_lush_matrix_file_name');
   --
   -- Inputs:
   --   filename : the name of the lush matrix file. (string)
   --
   -- Outputs:
   --   d   : matrix which is stored in 'filename'.
   --
   --   Koray Kavukcuoglu
   
   local fid = torch.DiskFile(filename,'r'):binary()
   local magic = fid:readInt()
   local ndim = fid:readInt()

   local tdims
   if ndim == 0 then
      tdims = torch.LongStorage({1})
   else
      tdims = fid:readInt(math.max(3,ndim))
   end
   local dims = torch.LongStorage(ndim)
   for i=1,ndim do dims[i] = tdims[i] end

   local nelem = 1
   for i=1,dims:size() do
      nelem = nelem * dims[i]
   end
   local d = torch.Storage()
   local x
   if magic == 507333717 then      --ubyte matrix
      d = fid:readByte(nelem)
      x = torch.ByteTensor(d,1,dims)
   elseif magic == 507333716 then      --integer matrix
      d = fid:readInt(nelem)
      x = torch.IntTensor(d,1,dims)
   elseif magic == 507333713 then      --float matrix
      d = fid:readFloat(nelem)
      x = torch.FloatTensor(d,1,dims)
   elseif magic == 507333715 then      --double matrix
      d = fid:readDouble(nelem)
      x = torch.DoubleTensor(d,1,dims)
   else
      error('Unknown magic number in binary lush matrix')
   end

   fid:close()
   return x
end

return lushio
