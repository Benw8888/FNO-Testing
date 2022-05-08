# Run both spectral and FNO methods and compare times

using Burgers
include("Burgers_spectral.jl")

x_resolution = 2^10  # number of data points the x axis is split into
dsamples = round(Int,2^13/x_resolution)  # how many other samples we pick from the base 2^13 dataset
data = Burgers.get_dataloader(Δsamples=dsamples)  # get data from Zongyi
_ ,test_data = data  # split data into train and test data
spectral_dt = 0.1

FNO_learner = Burgers.train(input_data=data, epochs=1)
FNO_model = FNO_learner.model
loss_fn = FNO_learner.lossfn

fno_time = 0
spectral_time = 0

fno_loss = 0
spectral_loss = 0
counter = 0

no_change_loss = 0

for (x,y) in test_data
    mini_batch_size = size(x)[3] # 100, 100, 5
    for i in 1:mini_batch_size
        counter = counter + 1
        print(counter)
        data_point = x[:,:,i:i]
        no_change_loss += loss_fn(data_point[2:2,:,:],y[:,:,i:i])

        dfno_time = @elapsed fno_res = FNO_model(data_point)
        fno_time += dfno_time

        dspectral_time = @elapsed spectral_res = spectral_output(n=x_resolution, input_data=data_point[2,:,1], viscosity=0.1, dt=spectral_dt)
        spectral_time += dspectral_time

        fno_loss += loss_fn(fno_res,y[:,:,i:i])
        spectral_loss += loss_fn(reshape(spectral_res,(1,x_resolution,1)),y[:,:,i:i])
    end
end

    # to get loss and time, divide fno_loss or spectral_loss and fno_time or spectral_time by counter
    # you need to divide by counter to get an average

println("PRINTING LOSSES")
println("FNO Average Loss: ",fno_loss/counter)
println("Spectral Time Iteration Average Loss: ",spectral_loss/counter)

println("FNO Average Time: ",fno_time/counter)
println("Spectral Time Iteration Average Time: ",spectral_time/counter)