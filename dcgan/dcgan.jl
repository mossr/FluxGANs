### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 9e2152fe-07f8-11eb-2181-0fc58c39abaa
begin
	using Base.Iterators: partition
	using CUDA
	using Flux
	using Flux.Optimise: update!
	using Flux.Losses: logitbinarycrossentropy
	using Images
	using MLDatasets
	using Statistics
	using Parameters
	using Printf
	using Random
	using PlutoUI
end

# ╔═╡ 89578430-07f8-11eb-0057-fb8e78c1252c
md"""
# Deep Convolutional GAN

Deep convolutional generative adversarial network (${\rm {\small DCGAN}}$).

- Source: modified from [Flux model-zoo](https://github.com/FluxML/model-zoo/blob/master/vision/dcgan_mnist/dcgan_mnist.jl)
> A. Radford, L. Metz, and S. Chintala, **"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"**, *International Conference on Learning Representations (ICLR)*, 2016. [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)
"""

# ╔═╡ 3c4efcc0-07fa-11eb-1a73-abc45b853fe3
md"""
## Hyperparameters
"""

# ╔═╡ a10d9d80-07f8-11eb-2512-3d6f569834c4
@with_kw struct HyperParameters
	batch_size::Int = 128
	latent_dim::Int = 100
	epochs::Int = 20
	verbose_freq::Int = 1000
	output_x::Int = 6
	output_y::Int = 6
	αᴰ::Float64 = 0.0002 # discriminator learning rate
	αᴳ::Float64 = 0.0002 # generator learning rate
end

# ╔═╡ a8d56d70-07fa-11eb-1d5a-6b6320e83f1e
md"""
## Sampling Generator
"""

# ╔═╡ e02c1a90-07f9-11eb-29d8-87f14c0acf9a
function create_output_image(gen, fixed_noise, hparams)
	@eval Flux.istraining() = false
	fake_images = cpu.(gen.(fixed_noise))
	@eval Flux.istraining() = true
	image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y))); dims=(3,4)), (2,1))
	image_array = Gray.(image_array .+ 1f0) ./ 2f0
	return image_array
end

# ╔═╡ a36b51b0-07fa-11eb-2c84-bf16c907739e
md"""
## Discriminator Network
"""

# ╔═╡ bba0d430-07fa-11eb-32d9-8d093bb0d3f0
function Discriminator()
	return Chain(
		Conv((4,4), 1=>64; stride=2, pad=1),
		x->leakyrelu.(x, 0.2f0),
		Dropout(0.25),
		Conv((4,4), 64=>128; stride=2, pad=1),
		x->leakyrelu.(x, 0.2f0),
		Dropout(0.25),
		x->reshape(x, 7*7*128, :),
		Dense(7*7*128, 1))
end

# ╔═╡ efb706e0-07fa-11eb-3363-630a62378a44
md"""
## Generator Network
"""

# ╔═╡ ff26e690-07fa-11eb-0d3c-5fa69b6c1b27
function Generator(hparams)
	return Chain(
		Dense(hparams.latent_dim, 7*7*256),
		BatchNorm(7*7*256, relu),
		x->reshape(x, 7, 7, 256, :),
		ConvTranspose((5,5), 256=>128; stride=1, pad=2),
		BatchNorm(128, relu),
		ConvTranspose((4,4), 128=>64; stride=2, pad=1),
		BatchNorm(64, relu),
		ConvTranspose((4,4), 64=>1, tanh; stride=2, pad=1))
end

# ╔═╡ 40070642-07fb-11eb-2283-a9d97a1d387c
md"""
## Loss functions
"""

# ╔═╡ afc54960-07fb-11eb-20e3-0f3d7474f5aa
md"""
### Discriminator loss
"""

# ╔═╡ 42d47e20-07fb-11eb-05ab-2f937c887bbc
function discriminator_loss(real_output, fake_output)
	real_loss = mean(logitbinarycrossentropy(real_output, 1f0, agg=identity))
	fake_loss = mean(logitbinarycrossentropy(fake_output, 0f0, agg=identity))
	return real_loss + fake_loss
end

# ╔═╡ ac0ad970-07fb-11eb-3e6c-f5aab45989c3
md"""
### Generator loss
"""

# ╔═╡ bb036640-07fb-11eb-3d27-2bd43ecbcb23
generator_loss(fake_output) = mean(logitbinarycrossentropy(fake_output, 1f0,
		                                                   agg=identity))

# ╔═╡ cdab1860-07fb-11eb-0ceb-795b03785886
md"""
## Training
"""

# ╔═╡ cf5bd5a0-07fb-11eb-044c-51af982820b9
function train_discriminator!(G, D, x, optD, hparams)
	noise = randn!(similar(x, (hparams.latent_dim, hparams.batch_size)))
	fake_input = G(noise)
	θ = Flux.params(D)
	loss, back = Flux.pullback(θ) do
		discriminator_loss(D(x), D(fake_input))
	end
	grad = back(1f0)
	update!(optD, θ, grad)
	return loss
end

# ╔═╡ 1f30b190-07fc-11eb-3d53-97bde80387a5
function train_generator!(G, D, x, optG, hparams)
	noise = randn!(similar(x, (hparams.latent_dim, hparams.batch_size)))
	θ = Flux.params(G)
	loss, back = Flux.pullback(θ) do
		generator_loss(D(G(noise)))
	end
	grad = back(1f0)
	update!(optG, θ, grad)
	return loss
end

# ╔═╡ 49c54470-07fc-11eb-38e9-034aae7f2037
function train(; kwargs...)
	# Model parameters
	hparams = HyperParameters(; kwargs...)

	# Load MNIST dataset
	images, _ = MLDatasets.MNIST.traindata(Float32)
	# Normalize to [-1, 1]
	image_tensor = reshape((2f0 .* images .- 1f0), 28, 28, 1, :)
	# Partition into batches
	partitions = partition(1:60_000, hparams.batch_size)
	data = [image_tensor[:, :, :, r] |> gpu for r in partitions]

	noise_length = hparams.output_x * hparams.output_y
	fixed_noise = [randn(hparams.latent_dim, 1) |> gpu for _ in 1:noise_length]

	# Networks
	D = Discriminator() |> gpu
	G = Generator(hparams) |> gpu

	# Optimizers
	optD = ADAM(hparams.αᴰ)
	optG = ADAM(hparams.αᴳ)

	# Training
	step = 0
	for epoch in 1:hparams.epochs
		@info "Epoch $epoch"
		for x in data
			# Update discriminator and generator
			loss_D = train_discriminator!(G, D, x, optD, hparams)
			loss_G = train_generator!(G, D, x, optG, hparams)

			# Logging
			if step % hparams.verbose_freq == 0
				@info "[$step] Discriminator loss = $loss_D, Generator loss = $loss_G"
				# Save generated fake image
				output_image = create_output_image(G, fixed_noise, hparams)
				save(@sprintf("output/dcgan_%06d.png", step), output_image)
			end
			step += 1
		end
	end

	output_image = create_output_image(G, fixed_noise, hparams)
	save(@sprintf("output/dcgan_%06d.png", step), output_image)
	return G, output_image
end

# ╔═╡ 6d37dcf0-07fd-11eb-0a58-31cac7537e26
md"""
## Running model
See console windows for mid-training output.
"""

# ╔═╡ e14be5a0-07fd-11eb-04e9-1746a9b678c1
# G, output = train()

# ╔═╡ e3c77c4c-0873-11eb-010a-99e55ea8fe1f
md"---"

# ╔═╡ b7b53414-0873-11eb-07c0-17029ef7a6c1
PlutoUI.TableOfContents("Deep Convolutional GAN")

# ╔═╡ Cell order:
# ╟─89578430-07f8-11eb-0057-fb8e78c1252c
# ╠═9e2152fe-07f8-11eb-2181-0fc58c39abaa
# ╟─3c4efcc0-07fa-11eb-1a73-abc45b853fe3
# ╠═a10d9d80-07f8-11eb-2512-3d6f569834c4
# ╟─a8d56d70-07fa-11eb-1d5a-6b6320e83f1e
# ╠═e02c1a90-07f9-11eb-29d8-87f14c0acf9a
# ╟─a36b51b0-07fa-11eb-2c84-bf16c907739e
# ╠═bba0d430-07fa-11eb-32d9-8d093bb0d3f0
# ╟─efb706e0-07fa-11eb-3363-630a62378a44
# ╠═ff26e690-07fa-11eb-0d3c-5fa69b6c1b27
# ╟─40070642-07fb-11eb-2283-a9d97a1d387c
# ╟─afc54960-07fb-11eb-20e3-0f3d7474f5aa
# ╠═42d47e20-07fb-11eb-05ab-2f937c887bbc
# ╟─ac0ad970-07fb-11eb-3e6c-f5aab45989c3
# ╠═bb036640-07fb-11eb-3d27-2bd43ecbcb23
# ╟─cdab1860-07fb-11eb-0ceb-795b03785886
# ╠═cf5bd5a0-07fb-11eb-044c-51af982820b9
# ╠═1f30b190-07fc-11eb-3d53-97bde80387a5
# ╠═49c54470-07fc-11eb-38e9-034aae7f2037
# ╟─6d37dcf0-07fd-11eb-0a58-31cac7537e26
# ╠═e14be5a0-07fd-11eb-04e9-1746a9b678c1
# ╟─e3c77c4c-0873-11eb-010a-99e55ea8fe1f
# ╠═b7b53414-0873-11eb-07c0-17029ef7a6c1
