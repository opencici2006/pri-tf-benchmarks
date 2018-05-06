# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

results_dir=$PWD/result

rm -rf $PWD/result/
mkdir -p $PWD/result


# Beginning of outer loop.
# ===============================================

for model in resnet50 alexnet googlenet vgg11 inception3
do
	echo "Working With Model : " $model
        output_filename="output_$model"
        echo "output filename:" $output_filename
	echo "****************************************************************************************************************"

        # Beginning of inner loop.	
	# ==============================================

	for curr_iteration in 1 2 3 4 5
	do
		echo "Working with Model :" $model "and Iteration : " $curr_iteration 
		output_filename_final=$output_filename
                output_filename_final+="_$curr_iteration.txt"
		echo "output filename: " $output_filename_final
		echo "---------------------------------------------------------------"

		python run_single_node_benchmark.py -c skl -m $model > $results_dir/$output_filename_final
		
	done 
 
	# End of inner loop.
	# ===============================================
done

# End of outer loop.
# ===============================================


echo "*********************************************************"
echo "All the models and Iterations are Completed."
echo "Please check your results in the directory: " $results_dir
