using CSV, DataFrames, Dates
source_data = DataFrame(CSV.File("2021Toronto_HumanBehaviourModels/CityofToronto_COVID-19_RecoveryData.csv"))
data_on = source_data[:, "7 day moving average"]
data_on_acc = [sum(data_on[1:i]) for i = 1: length(data_on)]
province_name, n, m, popu = "Toronto", 0,154,2930000
data_daily = data_on
data_acc = data_on_acc
start_data = Date(2020, 03, 01)
xlabel_str = "Days after $start_data"
source_data_label_daily = "Daily Confirmed Cases"
source_data_title_daily = string("Daily Confirmed Cases of ", province_name)
source_data_savefig_daily = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", province_name, "_daily_cases.png")
source_data_label_acc = "Accumulated Confirmed Cases"
source_data_title_acc = string("Accumulated Cases of ", province_name)
source_data_savefig_acc = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", province_name, "_accumulated_cases.png")
# Save the Source data
display(plot(data_daily, label=source_data_label_daily, xlabel=xlabel_str, title=source_data_title_daily, lw=2))
savefig(source_data_savefig_daily)
display(plot(data_acc, label=source_data_label_acc, xlabel=xlabel_str, title=source_data_title_acc, lw=2))
savefig(source_data_savefig_acc)
data_daily[1]
println(length(data_acc))
#plot(data_on[155:455])


# First Wave Data
savetitle = "First_wave"
province_name, n, m, popu = "Toronto", 0,154,2930000
start_data = Date(2020, 03, 01) + Dates.Day(n)
data_daily = data_on[(n + 1):(n + m +1)]
data_acc = data_on_acc[(n + 1):(n + m + 1)]
xlabel_str = "Days after $start_data"
source_data_label_daily = "Daily Confirmed Cases"
source_data_title_daily = string("Daily Confirmed Cases of ", province_name)
source_data_savefig_daily = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", savetitle,province_name, "_daily_cases.png")
source_data_label_acc = "Accumulated Confirmed Cases"
source_data_title_acc = string("Accumulated Cases of ", province_name)
source_data_savefig_acc = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", savetitle, province_name, "_accumulated_cases.png")
# Save the Source data
display(plot(data_daily, label=source_data_label_daily, xlabel=xlabel_str, title=source_data_title_daily, lw=2))
savefig(source_data_savefig_daily)
display(plot(data_acc, label=source_data_label_acc, xlabel=xlabel_str, title=source_data_title_acc, lw=2))
savefig(source_data_savefig_acc)
data_daily[1]
println(length(data_acc))

first_wave_data = DataFrame(Daily = data_daily, Acc = data_acc)

CSV.write("2021Toronto_HumanBehaviourModels/first_wave_data.csv", first_wave_data)

# Second Wave Data
savetitle = "Second_wave"
province_name, n, m, popu = "Toronto", 155,199,2930000
start_data = Date(2020, 03, 01) + Dates.Day(n)
data_daily = data_on[(n + 1):(n + m +1)]
data_acc = data_on_acc[(n + 1):(n + m + 1)]
xlabel_str = "Days after $start_data"
source_data_label_daily = "Daily Confirmed Cases"
source_data_title_daily = string("Daily Confirmed Cases of ", province_name)
source_data_savefig_daily = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", savetitle,province_name, "_daily_cases.png")
source_data_label_acc = "Accumulated Confirmed Cases"
source_data_title_acc = string("Accumulated Cases of ", province_name)
source_data_savefig_acc = string("./2021Toronto_HumanBehaviourModels/Saving_Data/", province_name, "/", savetitle, province_name, "_accumulated_cases.png")
# Save the Source data
display(plot(data_daily, label=source_data_label_daily, xlabel=xlabel_str, title=source_data_title_daily, lw=2))
savefig(source_data_savefig_daily)
display(plot(data_acc, label=source_data_label_acc, xlabel=xlabel_str, title=source_data_title_acc, lw=2))
savefig(source_data_savefig_acc)
data_daily[1]
println(length(data_acc))

second_wave_data = DataFrame(Daily = data_daily, Acc = data_acc)

CSV.write("2021Toronto_HumanBehaviourModels/second_wave_data.csv", second_wave_data)

