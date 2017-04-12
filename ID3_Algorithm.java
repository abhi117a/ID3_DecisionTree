
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

/*
 * This class is responsible for creating decision tree
 * author - Abhimanyu Rana(akr140230)
 * 		  
 */
public class ID3_Algorithm {
	private static int count = 0;


	public static void main(String[] args) {
		if (args.length != 3) {
			System.out.println("usage : path of training dataset, path of validation data set, path of test data set");
			return;
		}


		int[] featuresAndLength = find_features(args[0]);
		int[][] values = new int[featuresAndLength[1]][featuresAndLength[0]];
		String[] featureNames = new String[featuresAndLength[0]];
		int[] isDone = new int[featuresAndLength[0]];
		int[] indexList = new int[values.length];
		load_values(args[0], values, featureNames, isDone, indexList, featuresAndLength[0]);
		TreeNode root = construct_decision_tree(null, values, isDone, featuresAndLength[0] - 1, indexList, null);
		
		
		TreeNode pruneTree = post_prune_algorithm(args[1], 10, 10, root, values, featuresAndLength[0] - 1);
		System.out.println("-------------------------------------------------------------------------");
		System.out.println("Number of training instances						=	" + values.length);
		System.out.println("Number of training attributes						=	" + featureNames.length);
		System.out.println("Total number of nodes in the tree 					=	" + dfs(root));
		System.out.println("Total number of leaf nodes in the tree 				=	" + no_of_leaf_nodes(root));
		System.out.println("The Accuracy over Training data for decision Tree 	=	" + calculate_accuracy(args[0], root));

		System.out.println("Number of testing instances							=	" + load_testing_data(args[2]).length);
		System.out.println("Number of testing attributes 						=	" + load_testing_data(args[2])[0].length);

		System.out.println("The Accuracy over Tesing data for decision Tree 	=	" + calculate_accuracy(args[2], root));

		System.out.println("-------------------------------------------------------------------------");
		
		System.out.println("Number of training instances						=	" + values.length);
		System.out.println("Number of training attributes						=	" + featureNames.length);
		System.out.println("Total number of nodes in the tree 					=	" + dfs(pruneTree));
		System.out.println("Total number of leaf nodes in the tree 				=	" + no_of_leaf_nodes(pruneTree));
		System.out.println("The Accuracy over Training data for decision Tree 	=	" + calculate_accuracy(args[0], pruneTree));

		System.out.println("Number of testing instances							=	" + load_testing_data(args[2]).length);
		System.out.println("Number of testing attributes 						=	" + load_testing_data(args[2])[0].length);

		System.out.println("The Accuracy over Tesing data for decision Tree 	=	" + calculate_accuracy(args[2], pruneTree));
				
		System.out.println("-------------------------------------------------------------------------");
		System.out.println("Decision Tree = ---------------------------------------------------------");
		System.out.println("-------------------------------------------------------------------------");
		print_tree(root, 0, featureNames);
		System.out.println("-------------------------------------------------------------------------");
		System.out.println("Pruned Decision Tree = --------------------------------------------------");
		System.out.println("-------------------------------------------------------------------------");
		print_tree(pruneTree, 0, featureNames);
		System.out.println("-------------------------------------------------------------------------");

	}
	
	private static double calculate_Log(double fraction) {
		return Math.log10(fraction) / Math.log10(2);
	}

	private static TreeNode construct_node(TreeNode root, int[][] values, int[] isCompleted, int features,
			int[] indexList) {
		int i = 0;
		int j = 0;
		int k = 0;
		double max_Info_Gain = 0;
		int max_Left_Index[] = null;
		int max_Right_Index[] = null;
		int max_Index = -1;
		for (; i < features; i++) {
			if (isCompleted[i] == 0) {

				int[] leftIndex = new int[values.length];
				int[] rightIndex = new int[values.length];
				double entrophy = 0;
				double rightPositives = 0;
				double infoGain = 0;
				double rightNegatives = 0, leftPositives = 0, leftNegatives = 0;
				double negatives = 0;
				double positives = 0;
				double left = 0;
				double right = 0;
				double leftEntrophy = 0;
				double rightEntrophy = 0;
				for (k = 0; k < indexList.length; k++) {
					if (values[indexList[k]][features] == 1) {
						positives++;
					} else {
						negatives++;
					}
					if (values[indexList[k]][i] == 1) {
						rightIndex[(int) right++] = indexList[k];
						if (values[indexList[k]][features] == 1) {
							rightPositives++;
						} else {
							rightNegatives++;
						}

					} else {
						leftIndex[(int) left++] = indexList[k];
						if (values[indexList[k]][features] == 1) {
							leftPositives++;
						} else {
							leftNegatives++;
						}

					}

				}

				entrophy = (-1 * calculate_Log(positives / indexList.length) * ((positives / indexList.length)))
						+ (-1 * calculate_Log(negatives / indexList.length) * (negatives / indexList.length));
				leftEntrophy = (-1 * calculate_Log(leftPositives / (leftPositives + leftNegatives))
						* (leftPositives / (leftPositives + leftNegatives)))
						+ (-1 * calculate_Log(leftNegatives / (leftPositives + leftNegatives))
								* (leftNegatives / (leftPositives + leftNegatives)));
				rightEntrophy = (-1 * calculate_Log(rightPositives / (rightPositives + rightNegatives))
						* (rightPositives / (rightPositives + rightNegatives)))
						+ (-1 * calculate_Log(rightNegatives / (rightPositives + rightNegatives))
								* (rightNegatives / (rightPositives + rightNegatives)));
				if (Double.compare(Double.NaN, entrophy) == 0) {
					entrophy = 0;
				}
				if (Double.compare(Double.NaN, leftEntrophy) == 0) {
					leftEntrophy = 0;
				}
				if (Double.compare(Double.NaN, rightEntrophy) == 0) {
					rightEntrophy = 0;
				}

				infoGain = entrophy
						- ((left / (left + right) * leftEntrophy) + (right / (left + right) * rightEntrophy));
				if (infoGain >= max_Info_Gain) {
					max_Info_Gain = infoGain;
					max_Index = i;
					int leftTempArray[] = new int[(int) left];
					for (int index = 0; index < left; index++) {
						leftTempArray[index] = leftIndex[index];
					}
					int rightTempArray[] = new int[(int) right];
					for (int index = 0; index < right; index++) {
						rightTempArray[index] = rightIndex[index];
					}
					max_Left_Index = leftTempArray;
					max_Right_Index = rightTempArray;

				}
			}
		}
		root.targetAttribute = max_Index;
		root.leftIndices = max_Left_Index;
		root.rightIndices = max_Right_Index;
		return root;
	}


	public static boolean check_if_all_positives(int[] indexList, int[][] values, int features) {
		boolean oneOnly = true;
		for (int i : indexList) {
			if (values[i][features] == 0)
				oneOnly = false;
		}
		return oneOnly;

	}

	public static boolean check_if_all_negatives(int[] indexList, int[][] values, int features) {
		boolean zeroOnly = true;
		for (int i : indexList) {
			if (values[i][features] == 1)
				zeroOnly = false;
		}
		return zeroOnly;

	}

	public static int find_max(TreeNode root, int[][] values, int features) {
		int noOfOnes = 0;
		int noOfZeroes = 0;
		if (root.parent == null) {
			int i = 0;
			for (i = 0; i < values.length; i++) {
				if (values[i][features] == 1) {
					noOfOnes++;
				} else {
					noOfZeroes++;
				}
			}
		} else {
			for (int i : root.parent.leftIndices) {
				if (values[i][features] == 1) {
					noOfOnes++;
				} else {
					noOfZeroes++;
				}
			}

			for (int i : root.parent.rightIndices) {
				if (values[i][features] == 1) {
					noOfOnes++;
				} else {
					noOfZeroes++;
				}
			}
		}
		return noOfZeroes > noOfOnes ? 0 : 1;

	}

	public static boolean check_if_all_attributes_processed(int[] isDone) {
		boolean allDone = true;
		for (int i : isDone) {
			if (i == 0)
				allDone = false;
		}
		return allDone;
	}

	public static TreeNode construct_decision_tree(TreeNode root, int[][] values, int[] isDone, int features,
			int[] indexList, TreeNode parent) {
		if (root == null) {
			root = new TreeNode();
			if (indexList == null || indexList.length == 0) {
				root.label_no = find_max(root, values, features);
				root.isLeaf = true;
				return root;
			}
			if (check_if_all_positives(indexList, values, features)) {
				root.label_no = 1;
				root.isLeaf = true;
				return root;
			}
			if (check_if_all_negatives(indexList, values, features)) {
				root.label_no = 0;
				root.isLeaf = true;
				return root;
			}
			if (features == 1 || check_if_all_attributes_processed(isDone)) {
				root.label_no = find_max(root, values, features);
				root.isLeaf = true;
				return root;
			}
		}
		root = construct_node(root, values, isDone, features, indexList);
		root.parent = parent;
		if (root.targetAttribute != -1)
			isDone[root.targetAttribute] = 1;
		int leftIsDone[] = new int[isDone.length];
		int rightIsDone[] = new int[isDone.length];
		for (int j = 0; j < isDone.length; j++) {
			leftIsDone[j] = isDone[j];
			rightIsDone[j] = isDone[j];

		}

		root.left = construct_decision_tree(root.left, values, leftIsDone, features, root.leftIndices, root);
		root.right = construct_decision_tree(root.right, values, rightIsDone, features, root.rightIndices, root);
		return root;
	}

	public static void print_tree(TreeNode tree) {
		if (tree != null) {
			System.out.println("tree.targetAttribute " + tree.targetAttribute);
			System.out.println("tree.label " + tree.label_no);
			System.out.println("tree.isLeaf " + tree.isLeaf);
			if (tree.leftIndices != null) {
				System.out.println("tree.leftIndices ");
				for (int i : tree.leftIndices) {
					System.out.print(i + " ");
				}
			}
			if (tree.rightIndices != null) {
				System.out.println("\ntree.rightIndices ");
				for (int i : tree.rightIndices) {
					System.out.print(i + " ");
				}
			}
			System.out.println();
			print_tree(tree.left);
			print_tree(tree.right);
		}
	}


	public static TreeNode create_copy(TreeNode root) {
		if (root == null)
			return root;

		TreeNode temp = new TreeNode();
		temp.label_no = root.label_no;
		temp.isLeaf = root.isLeaf;
		temp.leftIndices = root.leftIndices;
		temp.rightIndices = root.rightIndices;
		temp.targetAttribute = root.targetAttribute;
		temp.parent = root.parent;
		temp.left = create_copy(root.left); 
		temp.right = create_copy(root.right); 
		return temp;
	}


	public static TreeNode post_prune_algorithm(String pathName, int L, int K, TreeNode root, int[][] values,
			int features) {
		TreeNode postPrunedTree = new TreeNode();
		int i = 0;
		postPrunedTree = root;
		double maxAccuracy = measure_accuracy_over_validation_set(pathName, root);
		for (i = 0; i < L; i++) {
			TreeNode newRoot = create_copy(root);
			Random randomNumbers = new Random();
			int M = 1 + randomNumbers.nextInt(K);
			for (int j = 1; j <= M; j++) {
				count = 0;
				int noOfNonLeafNodes = find_number_of_non_leaf_nodes(newRoot);
				if (noOfNonLeafNodes == 0)
					break;
				TreeNode nodeArray[] = new TreeNode[noOfNonLeafNodes];
				build_array(newRoot, nodeArray);
				int P = randomNumbers.nextInt(noOfNonLeafNodes);
				nodeArray[P] = create_leaf_with_majority_elements(nodeArray[P], values, features);

			}
			double accuracy = measure_accuracy_over_validation_set(pathName, newRoot);

			if (accuracy > maxAccuracy) {
				postPrunedTree = newRoot;
				maxAccuracy = accuracy;
			}
		}
		return postPrunedTree;
	}

	private static double measure_accuracy_over_validation_set(String pathName, TreeNode newRoot) {
		int[][] validationSet = construct_validation_set(pathName);
		double count = 0;
		for (int i = 1; i < validationSet.length; i++) {
			count += check_if_correctly_classified(validationSet[i], newRoot);
		}
		return count / validationSet.length;
	}


	private static int check_if_correctly_classified(int[] setValues, TreeNode newRoot) {
		int index = newRoot.targetAttribute;
		int correctlyClassified = 0;
		TreeNode testingNode = newRoot;
		while (testingNode.label_no == -1 && index >= 0) {
			if (setValues[index] == 1) {
				testingNode = testingNode.right;
			} else {
				testingNode = testingNode.left;
			}
			if (testingNode.label_no == 1 || testingNode.label_no == 0) {
				if (setValues[setValues.length - 1] == testingNode.label_no) {
					correctlyClassified = 1;
					break;
				} else {
					break;
				}
			}
			index = testingNode.targetAttribute;
		}
		return correctlyClassified;
	}

	private static int[][] construct_validation_set(String pathName) {
		int[] featuresAndLength = find_features(pathName);
		String csvFile = pathName;
		int[][] validationSet = new int[featuresAndLength[1]][featuresAndLength[0]];
		BufferedReader bufferedReader = null;
		String line = "";
		String cvsSplitBy = "\t";
		try {
			bufferedReader = new BufferedReader(new FileReader(csvFile));
			int i = 0;
			int count = 0;
			while ((line = bufferedReader.readLine()) != null) {
				String[] lineParameters = line.split(cvsSplitBy);
				int j = 0;
				if (count == 0) {
					count++;
					continue;
				} else {
					for (String lineParameter : lineParameters) {
						validationSet[i][j++] = Integer.parseInt(lineParameter);
					}
				}
				i++;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (bufferedReader != null) {
				try {
					bufferedReader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return validationSet;
	}

	private static int find_max_at_given_node(TreeNode root, int[][] values, int features) {
		int noOfOnes = 0;
		int noOfZeroes = 0;
		if (root.leftIndices != null) {
			for (int i : root.leftIndices) {
				if (values[i][features] == 1) {
					noOfOnes++;
				} else {
					noOfZeroes++;
				}
			}
		}

		if (root.rightIndices != null) {
			for (int i : root.rightIndices) {
				if (values[i][features] == 1) {
					noOfOnes++;
				} else {
					noOfZeroes++;
				}
			}
		}
		return noOfZeroes > noOfOnes ? 0 : 1;
	}

	private static TreeNode create_leaf_with_majority_elements(TreeNode node, int[][] values, int features) {
		node.isLeaf = true;
		node.label_no = find_max_at_given_node(node, values, features);
		node.left = null;
		node.right = null;
		return node;
	}

	private static void build_array(TreeNode root, TreeNode[] nodeArray) {
		if (root == null || root.isLeaf) {
			return;
		}
		nodeArray[count++] = root;
		if (root.left != null) {
			build_array(root.left, nodeArray);
		}
		if (root.right != null) {
			build_array(root.right, nodeArray);
		}
	}

	private static int find_number_of_non_leaf_nodes(TreeNode root) {
		if (root == null || root.isLeaf)
			return 0;
		else
			return (1 + find_number_of_non_leaf_nodes(root.left) + find_number_of_non_leaf_nodes(root.right));
	}


	private static int dfs(TreeNode root) {
		if (root == null) {
			return 0;
		} else {
			return 1 + dfs(root.left) + dfs(root.right);
		}

	}

	private static int no_of_leaf_nodes(TreeNode root) {
		if (root == null) {
			return 0;
		}
		if (root.left == null && root.right == null) {
			return 1;
		} else {
			return dfs(root.left) + dfs(root.right);
		}

	}

	private static int[] find_features(String csvFile) {
		BufferedReader bufferedReader = null;
		String line = "";
		String cvsSplitBy = "\t";
		int features = 0;
		int count = 0;
		int[] featuresAndLength = new int[2];
		try {

			bufferedReader = new BufferedReader(new FileReader(csvFile));
			while ((line = bufferedReader.readLine()) != null) {
				if (count == 0) {
					String[] country = line.split(cvsSplitBy);
					featuresAndLength[0] = country.length;
				}
				count++;
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (bufferedReader != null) {
				try {
					bufferedReader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		featuresAndLength[1] = count;
		return featuresAndLength;
	}

	private static void load_values(String pathName, int[][] values, String[] featureNames, int[] isDone,
			int[] indexList, int features) {
		String csvFile = pathName;
		BufferedReader bufferedReader = null;
		String line = "";
		String cvsSplitBy = "\t";
		for (int k = 0; k < features; k++) {
			isDone[k] = 0;
		}
		int k = 0;
		for (k = 0; k < values.length; k++) {
			indexList[k] = k;
		}
		try {

			bufferedReader = new BufferedReader(new FileReader(csvFile));
			int i = 0;
			while ((line = bufferedReader.readLine()) != null) {
				String[] lineParameters = line.split(cvsSplitBy);
				int j = 0;
				if (i == 0) {
					for (String lineParameter : lineParameters) {
						featureNames[j++] = lineParameter;
					}
				}

				else {

					for (String lineParameter : lineParameters) {
						values[i][j++] = Integer.parseInt(lineParameter);
					}
				}
				i++;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (bufferedReader != null) {
				try {
					bufferedReader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private static void print_tree(TreeNode root, int printLines, String[] featureNames) {
		int printLinesForThisLoop = printLines;
		if (root.isLeaf) {
			System.out.println(" " + root.label_no);
			return;
		}
		for (int i = 0; i < printLinesForThisLoop; i++) {
			System.out.print("| ");
		}
		if (root.left != null && root.left.isLeaf && root.targetAttribute != -1)
			System.out.print(featureNames[root.targetAttribute] + "= 0 :");
		else if (root.targetAttribute != -1)
			System.out.println(featureNames[root.targetAttribute] + "= 0 :");

		printLines++;
		print_tree(root.left, printLines, featureNames);
		for (int i = 0; i < printLinesForThisLoop; i++) {
			System.out.print("| ");
		}
		if (root.right != null && root.right.isLeaf && root.targetAttribute != -1)
			System.out.print(featureNames[root.targetAttribute] + "= 1 :");
		else if (root.targetAttribute != -1)
			System.out.println(featureNames[root.targetAttribute] + "= 1 :");
		print_tree(root.right, printLines, featureNames);
	}


	private static double calculate_accuracy(String pathName, TreeNode root) {
		double accuracy = 0;
		int[][] testingData = load_testing_data(pathName);
		for (int i = 0; i < testingData.length; i++) {
			accuracy += check_if_correctly_classified(testingData[i], root);
		}
		return accuracy / testingData.length;

	}


	private static int[][] load_testing_data(String pathName) {
		int[] featuresAndLength = find_features(pathName);
		String csvFile = pathName;
		int[][] validationSet = new int[featuresAndLength[1]][featuresAndLength[0]];
		BufferedReader bufferReader = null;
		String line = "";
		String cvsSplitBy = "\t";
		try {

			bufferReader = new BufferedReader(new FileReader(csvFile));
			int i = 0;
			int count = 0;
			while ((line = bufferReader.readLine()) != null) {
				String[] lineParameters = line.split(cvsSplitBy);
				int j = 0;
				if (count == 0) {
					count++;
					continue;
				}

				else {

					for (String lineParameter : lineParameters) {
						validationSet[i][j++] = Integer.parseInt(lineParameter);
					}
				}
				i++;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (bufferReader != null) {
				try {
					bufferReader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return validationSet;
	}
}

class TreeNode {
	int label_no = -1;
	boolean isLeaf = false;
	int targetAttribute = -1;
	TreeNode parent;
	TreeNode left;
	TreeNode right;
	int leftIndices[];
	int rightIndices[];
}