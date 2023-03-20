import numpy as np
from random import randrange

def evaluate_student_performance(student, metrics_list, training_X, training_y, validate_X, validate_y):
    
    """
    Evaluate the performance of students
    
    Parameters
    ----------
    student: model
        the student model for evaluations
    metrics_list: dictionary
        contains the key(measurement name) and value (measurement function)
    training_X: numpy.array
        contains the training features for the student
    training_y: numpy.array
        contains the training outputs for the student
    validate_X: numpy.array
        contains the validate features for the student
    validate_y: numpy.array
        contains the validate outputs for the student
    
    Returns
    -------
    metrics_result: dictionary
        key is the name of performance metrics, and value is its performance values
    """  
    
    student_model = student.fit(training_X, training_y)
    predictions = student_model.predict(validate_X)
    
    metrics_result = {}
    for metric in metrics_list:
        metrics_result[metric[0]] = []
    
    for name, metric in metrics_list:
        metrics_result[name].append(metric(validate_y, predictions))
    return metrics_result


def pick_next_example(teacher, trainX, trainy, learnX, learny):
    
    """
    Pick the next data point for active learning
    
    Parameters
    ----------
    teacher: model
        the model to select the most uncertain points, and it indicates passive learning when input is None
    trainX: numpy.array
        contains the training features for the teacher
    trainy: numpy.array
        contains the training outputs for the teacher
    learnX: numpy.array
        contains the learning features for the teacher
    learny: numpy.array
        contains the learning outputs for the teacher
    
    Returns
    -------
    teacher: model
        the model to select the most uncertain points
    trainx: numpy.array
        updated training features for the teacher
    trainy: numpy.array
        updated training outputs for the teacher
    learnX: numpy.array
        updated learning features for the teacher
    learny: numpy.array
        updated learning outputs for the teacher
    """  
    
    # If it is passive, then randomly select a training point
    if teacher == None:
        new_pick = randrange(len(learny))
    else:
        teacher_model = teacher.fit(trainX, trainy)
        probas = teacher_model.predict_proba(learnX)
        new_pick = np.argmin(np.var(probas, axis=1))
        
    trainX = np.vstack((trainX, learnX[new_pick]))
    trainy = np.append(trainy, learny[new_pick])
    learnX = np.delete(learnX, new_pick, 0)
    learny = np.delete(learny, new_pick)
    
    return teacher, trainX, trainy, learnX, learny


def teach(teacher, students_list, data, featurizer, metrics_list, repeats, standardize):
    
    """
    Takes the teacher-student models and calcaultes their performance based on the input dataset and metrics
    
    Parameters
    ----------
    teacher: model
        the model to select the most uncertain points, and it indicates passive learning when input is None
    studens_list: list
        list contains students' names and their corresponding functions
    data: object
        dataset from https://tdcommons.ai
    featurizer: object
        function that converts molecular structures to integers/doubles (https://deepchem.io)
    metrics_list: list 
        list contains model performance metrics' names and their corresponding fuctions
    repeats: int
        integer that indicates how many random seeds used in the scaffold splitting
    standardize: boolean
        boolean that indicates if the training and validate datasets need to be standardized (e.g. rdkit descriptor)
    
    Returns
    -------
    students_result: dictionary
        key is the name of the stundent, and value contains another dictionary, which has the key of 
            performance metrics' names and value of list, which has iterations and the performance values
            druing each iteration (100 iterations in total by default: 10 random seeds for scaffold 
            split and 10 randoms for initialization)
    """   
    
    # initialize students_result(return) and metrics_results. Both are dictionary types.
    students_result = {}
    metrics_result = {}
    for student in students_list:
        students_result[student[0]] = []
        
    for random_seed in range (repeats):
        split = data.get_split(method = 'scaffold', frac = [0.5, 0.5, 0.0], seed=random_seed)
        
        # Go through # iterations (10 in the default cases)
        for index in range(10):

            learning_x = featurizer.featurize(split["train"].Drug)
            learning_y = split["train"].Y.to_numpy()
            validate_x = featurizer.featurize(split["valid"].Drug)
            validate_y = split["valid"].Y.to_numpy()

            if standardize == True:
                scaler = StandardScaler()
                learning_x = scaler.fit_transform(learning_x_)
                validate_x = scaler.transform(validate_x_)
                learning_x = np.nan_to_num(learning_x)
                validate_x = np.nan_to_num(validate_x)
            
            np.random.seed(index)
            training_mask = np.array([False for i in learning_y])
            training_mask[np.random.choice(np.where(learning_y == 0)[0])] = True
            training_mask[np.random.choice(np.where(learning_y == 1)[0])] = True

            training_x = learning_x[training_mask]
            training_y = learning_y[training_mask]
            learning_x = learning_x[np.invert(training_mask)]
            learning_y = learning_y[np.invert(training_mask)]

            # Initialize a nested dictionary to store performance metrics in training_size for one iteration (e.g. 6 * 222)  
            students_temp_result = {}
            for student in students_list:
                name = student[0]
                students_temp_result[name] = {}
                for metric in metrics_list:
                    metric_name = metric[0]
                    students_temp_result[name][metric_name] = []

            for i in range(len(learning_y)-2): 
                # Going through each datapoint for each student and calculate each metrics
                for name, model in students_list:
                    metrics_result = evaluate_student_performance(model, metrics_list,training_x,
                                                                  training_y, validate_x, validate_y)

                    for metric in metrics_list:
                        metric_name = metric[0]
                        students_temp_result[name][metric_name].append(metrics_result[metric_name])

                teacher, training_x, training_y, learning_x, learning_y = pick_next_example(teacher,
                                                                                            training_x,
                                                                                            training_y,
                                                                                            learning_x,
                                                                                            learning_y)
            # Attach the performance metrics after each iteration   
            for student in students_list:
                students_result[student[0]].append(students_temp_result[student[0]])
                
    return students_result
    




