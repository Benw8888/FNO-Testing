# Run both spectral and FNO methods and compare times

using Burgers
include("Burgers_spectral.jl")
data_array = Array{Float64}(undef, 7, 4)
for exponent=6:12
    FNO_learner = nothing
    FNO_model = nothing

    x_resolution = 2^(exponent) # number of data points the x axis is split into
    dsamples = round(Int,2^13/x_resolution)  # how many other samples we pick from the base 2^13 dataset
    data = Burgers.get_dataloader(Δsamples=dsamples)  # get data from Zongyi
    _ ,test_data = data  # split data into train and test data
    spectral_dt = 0.1

    FNO_learner = Burgers.train(input_data=data, epochs=100)
    FNO_model = cpu(FNO_learner.model)
    loss_fn = FNO_learner.lossfn

    fno_time = 0
    spectral_time = 0

    fno_loss = 0
    spectral_loss = 0
    counter = 0

    no_change_loss = 0

    for (x,y) in test_data
        mini_batch_size = size(x)[3] # 100, 100, 5
        for i in 1:5 #mini_batch_size
            counter = counter + 1
            println(counter)
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
    data_array[exponent-5,:] = [fno_loss/counter, spectral_loss/counter, fno_time/counter, spectral_time/counter]
    CUDA.reclaim()
end

FNO_loss = data_array[:,1]
Spectral_loss = data_array[:,2]
FNO_time = data_array[:,3]
Spectral_time = data_array[:,4]

plot(FNO_time, FNO_loss, label = "FNO")
plot!(Spectral_time, Spectral_loss, label = "Spectral Method", xlabel = "Time (s)", ylabel = "Loss", title = "Loss vs. Time")

x = [2^6, 2^7, 2^8, 2^9, 2^10, 2^11, 2^12]
plot(x, FNO_loss, label = "FNO", xaxis=:log)
plot!(x, Spectral_loss, label = "Spectral", xlabel = "Resolution", ylabel = "Loss", title = "Loss vs. Resolution", xaxis=:log)

plot(x, FNO_time, label = "FNO", xaxis=:log)
plot!(x, Spectral_loss, label = "Spectral", xlabel = "Resolution", ylabel = "Time (s)", title = "Time vs. Resolution", xaxis=:log)
