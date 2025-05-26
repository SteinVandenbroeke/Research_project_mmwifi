from model import *
import Mess_60Ghz
import Mess_5Ghz
from models.Mess_5Ghz.dataset import WifiCSIDataset as WifiCSIDataset5Ghz
from models.Mess_60Ghz.dataset import WifiCSIDataset as WifiCSIDataset60Ghz
from models.regression_model import RegressionTestModel

print(f"-- 5Ghz--")
dataset = WifiCSIDataset5Ghz('Mess_5Ghz/data/output_5Ghz.csv', 10, False, False)
default_test = TestModel("5Ghz_sample_size_test",dataset, num_epochs=100, dataset_split=[0.8, 0.1, 0.1], random_split=True)
training_size, validation_size, test_size = default_test.get_dataset_count()
default_test.train(False)
default_test.test(False)

# print(f"- regression -")
# default_test = RegressionTestModel("5Ghz_Regression_sample_size",dataset, learning_rate=0.0001, num_epochs=250, batch_size=1000, random_split=True)
# training_size, validation_size, test_size = default_test.get_dataset_count()
# for i in range(20):
#     cord = default_test.label_to_coord(i)
#     print(f"location {i}:", cord)
#     print(f"location {i} ==", default_test.coord_to_label([cord]))
# default_test.train(True)
# default_test.test(True)

#
print(f"-- 5Ghz--")
dataset = WifiCSIDataset5Ghz('Mess_5Ghz/data/output_5Ghz.csv', 10, False, False)
default_test = TestModel("5Ghz_sample_size_test_no_stein",dataset, num_epochs=100, dataset_split=[0.8, 0.1, 0.1], random_split=True, hidden_size=200)
dataset_removed_stein = WifiCSIDataset5Ghz('Mess_5Ghz/data/output_5Ghz.csv', 10, False, False, remove_names=["Stein"], name="no_stein_")
dataset_only_stein = WifiCSIDataset5Ghz('Mess_5Ghz/data/output_5Ghz.csv', 10, False, False, remove_names=["Andre","Daniel","Heng","Maksim","Peter","Sadegh","Wouter",'Andres maria',"David","Jacob","Matthias","Priyesh",'Xinlei liu',"Arno","Fabian","Lynn","Nabeel","Ruben","Wout"],name="every_body_except_stein")
default_test.training_dataset, default_test.validation_dataset = torch.utils.data.random_split(dataset_removed_stein, [0.8,0.2], torch.Generator().manual_seed(0))
default_test.test_dataset = dataset_only_stein
default_test.train(True)
default_test.test(True)

print(f"- regression -")
default_test = RegressionTestModel("5Ghz_Regression_sample_size",dataset, learning_rate=0.0001, num_epochs=250, batch_size=1000, random_split=True)
training_size, validation_size, test_size = default_test.get_dataset_count()
for i in range(20):
    cord = default_test.label_to_coord(i)
    print(f"location {i}:", cord)
    print(f"location {i} ==", default_test.coord_to_label([cord]))
default_test.train(False)
default_test.test(False)

exit()
#
#
# print(f"- regression - remove location")
# dataset_removed_11 = WifiCSIDataset5Ghz('Mess_5Ghz/data/output_5Ghz.csv', 30, False, False, remove_locations=[11], name="no_11_")
# dataset_only_11 = WifiCSIDataset5Ghz('Mess_5Ghz/data/output_5Ghz.csv', 30, False, False, remove_locations=[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19],name="only_11_")
# regresion_test = RegressionTestModel("5Ghz_Regression_sample_size_remove_pos_11",dataset_removed_11, learning_rate=0.0001, num_epochs=250, batch_size=1000, random_split=True)
# regresion_test.training_dataset, regresion_test.validation_dataset = torch.utils.data.random_split(dataset_removed_11, [0.8,0.2], torch.Generator().manual_seed(0))
# regresion_test.test_dataset = dataset_only_11
#
# training_size, validation_size, test_size = regresion_test.get_dataset_count()
# for i in range(20):
#     cord = regresion_test.label_to_coord(i)
#     print(f"location {i}:", cord)
#     print(f"location {i} ==", regresion_test.coord_to_label([cord]))
# regresion_test.train(False)
# regresion_test.test(False)


print(f"- regression - remove person")
dataset_removed_stein = WifiCSIDataset5Ghz('Mess_5Ghz/data/output_5Ghz.csv', 10, False, False, remove_names=["Stein"], name="no_stein_")
dataset_only_stein = WifiCSIDataset5Ghz('Mess_5Ghz/data/output_5Ghz.csv', 10, False, False, remove_names=["Andre","Daniel","Heng","Maksim","Peter","Sadegh","Wouter",'Andres maria',"David","Jacob","Matthias","Priyesh",'Xinlei liu',"Arno","Fabian","Lynn","Nabeel","Ruben","Wout"],name="every_body_except_stein")
regresion_test = RegressionTestModel("5Ghz_Regression_sample_size_remove_name_stein",dataset_removed_stein, learning_rate=0.0001, num_epochs=250, batch_size=1000, random_split=True)
regresion_test.training_dataset, regresion_test.validation_dataset = torch.utils.data.random_split(dataset_removed_stein, [0.8,0.2], torch.Generator().manual_seed(0))
regresion_test.test_dataset = dataset_only_stein

training_size, validation_size, test_size = regresion_test.get_dataset_count()
for i in range(20):
    cord = regresion_test.label_to_coord(i)
    print(f"location {i}:", cord)
    print(f"location {i} ==", regresion_test.coord_to_label([cord]))
regresion_test.train(True)
regresion_test.test(True)

#
exit()
print(f"-- 60Ghz--")
dataset = WifiCSIDataset60Ghz('Mess_60Ghz/data/60Ghz_position_data.csv', 1, False, False, subtract_background_noice=True)
default_test = TestModel("60Ghz_sample_size_new",dataset, num_epochs=800, random_split=True)
training_size, validation_size, test_size = default_test.get_dataset_count()
default_test.train(False)
default_test.test(False)

print(f"- regression -")
default_test = RegressionTestModel("60Ghz_Regression_sample_size",dataset, num_epochs=800, random_split=True)
training_size, validation_size, test_size = default_test.get_dataset_count()
default_test.train(False)
default_test.test(False)



# input()
# default_test.test(True)
#
