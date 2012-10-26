require 'mnist'
-- testing linear scaling of MINST to [0..1]
d = mnist.dataset({linscale01=true})
--d = mnist.dataset({original=true})

print(d[1]['input']:reshape(28,28))

