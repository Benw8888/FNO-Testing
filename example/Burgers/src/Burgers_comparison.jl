# Run both spectral and FNO methods and compare times

using Burgers
include("Burgers_spectral.jl")

x_resolution = 2^10  # number of data points the x axis is split into
dsamples = round(Int,2^13/x_resolution)  # how many other samples we pick from the base 2^13 dataset
data = Burgers.get_dataloader(Δsamples=dsamples)  # get data from Zongyi
_ ,test_data = data  # split data into train and test data

print(test_data)

FNO_learner = Burgers.train(input_data=data, epochs=1)
FNO_model = FNO_learner.model
loss_fn = FNO_learner.lossfn

fno_time = 0
spectral_time = 0

fno_loss = 0
spectral_loss = 0
counter = 0

for (x,y) in test_data
    mini_batch_size = size(x)[3] # 100, 100, 5
    for i in 1:mini_batch_size
        counter = counter + 1
        data_point = x[:,:,i:i]

        #fno_res = FNO_model(data_point)
        spectral_res = spectral_output(n=x_resolution, input_data=data_point[2,:,:], viscosity=0.1)

        #fno_time += @elapsed FNO_model(data_point)
        #spectral_time += @elapsed spectral_output(input_data=data_point[2,:], viscosity=0.1)

        #fno_loss += loss_fn(fno_res,y[:,:,i:i])
        #spectral_loss += loss_fn(spectral_res,y[:,:,i:i])
    end
end

    #counter = counter + 1
    #spectral_result = spectral_output(x,0.1)
    #spectral_time = spectral_time + @elapsed spec