import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import *
import csv
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
def initialize_population(pop_size, num_features):
    """
    Create an initial population of given size.
    Each individual is a binary mask array of length num_features.
    """
    population = []
    for _ in range(pop_size):
        # Randomly 0 or 1 for each feature
        individual = np.random.randint(2, size=num_features)
        population.append(individual)
    return population

def fitness_function(individual, X_train, y_train, X_val, y_val, X_val2, y_val2, coefficients=[0.429,0.286,0.143,0.071,0.071]):
    """
    Defines the fitness of an individual (feature mask).
    We train a DecisionTree on the selected features and measure accuracy on validation data.
    """
    # Identify which features are selected
    selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
    
    # Handle edge case: if no features are selected, return very low fitness
    if len(selected_indices) == 0:
        return 0.0
    
    # Subset the training and validation sets
    X_train_sel = X_train[:, selected_indices]
    X_val_sel   = X_val[:, selected_indices]
    X_val2_sel   = X_val2[:, selected_indices]
    num_features = X_train.shape[1]

    # Train the model on these features
    model = DecisionTreeClassifier(max_depth=10)
    #model = XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0.1,learning_rate=0.1,max_depth=5,min_child_weight=1,subsample=0.9)
    #model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf = 20,max_features='sqrt')
    model.fit(X_train_sel, y_train)
    
    ytrain_pred= model.predict(X_train_sel)
    y_pred = model.predict(X_val_sel)
    y_pred2 = model.predict(X_val2_sel)

    train_acc=accuracy_score(y_train, ytrain_pred)
    val_acc = accuracy_score(y_val, y_pred)
    #if val_acc<0.7:
        #return 0
    val2_acc = accuracy_score(y_val2, y_pred2)
    return coefficients[0]*val_acc+coefficients[1]*val2_acc+coefficients[2]*train_acc + coefficients[3]*(1-sum(individual)/num_features)+coefficients[4]*(1-abs(val2_acc-val_acc)/0.3)

def selection(population, fitnesses):
    """
    Tournament selection: pick two individuals at random, return the one with higher fitness.
    """
    idx1, idx2 = random.sample(range(len(population)), 2)
    if fitnesses[idx1] > fitnesses[idx2]:
        return population[idx1].copy()
    else:
        return population[idx2].copy()

def crossover(parent1, parent2):
    """
    One-point crossover.
    """
    # We could use a random crossover point
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

def crossover_old(parent1, parent2, n_flips=3):
    child1 = parent1 & parent2  # bitwise AND of 0/1 arrays
    child2 = child1.copy()
    candidate_indices = np.where((child1 == 0) & ((parent1 == 1) | (parent2 == 1)))[0]
    
    
    # Step 3: Randomly pick up to n_flips of those indices and flip them to 1
    if len(candidate_indices) > 0:
        chosen_indices1 = np.random.choice(candidate_indices, 
                                          size=min(n_flips, len(candidate_indices)), 
                                          replace=False)
        child1[chosen_indices1] = 1

        chosen_indices2 = np.random.choice(candidate_indices, 
                                          size=min(n_flips, len(candidate_indices)), 
                                          replace=False)
        child2[chosen_indices2] = 1

    return child1,child2
def mutate(individual, mutation_rate=0.01):
    """
    Flip bits with a certain mutation rate.
    """
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Flip bit (0 -> 1, 1 -> 0)
    return individual

def genetic_algorithm_feature_selection(X_train, y_train, X_val, y_val,X_val2, y_val2,
                                        pop_size=20, 
                                        generations=50,
                                        mutation_rate=0.01,
                                        elitism=True,coefficients=[0.429,0.286,0.143,0.071,0.071]):
    """
    Main GA loop:
      1) Initialize population
      2) Evaluate fitness
      3) Evolve population
      4) Return best solution found
    """
    num_features = X_train.shape[1]
    
    # 1) Initialize population
    population = initialize_population(pop_size, num_features)
    
    # Keep track of best solution
    best_individual = None
    best_fitness = 0.0
    
    for gen in range(generations):
        fitnesses = []
        top_pairs = []  # List to store (fitness, individual) pairs
        # 2) Calculate fitness for each individual
        """fitnesses = [fitness_function(ind, X_train, y_train, X_val, y_val, X_val2, y_val2,coefficients=coefficients) 
                     for ind in population]
        
        # Update best solution
        for ind, fit in zip(population, fitnesses):
            if fit >= best_fitness:
                best_fitness = fit
                best_individual = ind.copy()"""
        for ind in population:
            fit = fitness_function(ind, X_train, y_train, X_val, y_val, X_val2, y_val2, coefficients=coefficients)
            fitnesses.append(fit)
            if fit >= best_fitness:
                best_fitness = fit
                best_individual = ind.copy()
             # Add to top_three
            top_pairs.append((fit, ind.copy()))
        
        top_pairs.sort(reverse=True, key=lambda x: x[0])  # Assuming higher fitness is better
        # Print progress
        print(f"Generation {gen+1}/{generations} | Best Fitness so far: {best_fitness:.4f}")
        
        # 3) Evolve population
        new_population = []
        
        # If elitism is used, preserve the best individual directly
        if elitism:
            #new_population.append(best_individual.copy())
            for _, ind in top_pairs[:10]:  # Slice to get only the first 3 pairs
                new_population.append(ind.copy())
        
        # Fill the rest of the population
        while len(new_population) < pop_size:
            # Selection
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        population = new_population
    
    # Return the best solution found
    return best_individual, best_fitness

# ----------------------------------------------------------------------
# Usage Example (assuming you already have X_train, y_train, X_test, y_test, etc.)
# ----------------------------------------------------------------------
TARGET_COLUMN = 'to_buy_1d'
USE_STOCK_COLUMN = False
df = pd.read_csv('C:/Users/aziz8/Documents/tradingBot/clean.csv')
df = df.sample(frac=1).reset_index(drop=True)
df.drop_duplicates(inplace=True)
df = df.dropna()
df = df[df['Stock'].isin(['ORCL','HIMS','SHOP'])]
df = balance_binary_target(df, TARGET_COLUMN)
percentages = df[['to_buy_1d']].mean() * 100
print(percentages.round(2).to_string()) 
drop_columns = ['Stock', 'Date', 'Close', TARGET_COLUMN, 
                'to_buy_1d', 'to_buy_2d', 'to_buy_1d31', 
                'to_buy_intraday', "PM_max", "PM_min"]
X = df.drop(columns=drop_columns, errors='ignore')
df["Date"] = pd.to_datetime(df["Date"])
today = pd.to_datetime("today").normalize()
date1 = today - pd.Timedelta(days=300) #put 300
date1a = today - pd.Timedelta(days=100)
date2   = today - pd.Timedelta(days=50) 
mask_train = (df["Date"] >= date1) & (df["Date"] < date2)
mask_val = (df["Date"] >= date2) & (df["Date"] < today)
mask_val2 = (df["Date"] >= date1a) & (df["Date"] < today)
df_train = df[mask_train]
df_val = df[mask_val]
df_val2 = df[mask_val2]
X_train = df_train.drop(columns=drop_columns, errors='ignore')
y_train = df_train[TARGET_COLUMN]
X_val = df_val.drop(columns=drop_columns, errors='ignore')
y_val= df_val[TARGET_COLUMN]
X_val2 = df_val2.drop(columns=drop_columns, errors='ignore')
y_val2= df_val2[TARGET_COLUMN]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_val2_scaled = scaler.transform(X_val2)

X_train_scaled_df = pd.DataFrame(X_train_scaled, 
                                 columns=X_train.columns, 
                                 index=X_train.index)
X_val_scaled_df = pd.DataFrame(X_val_scaled, 
                                 columns=X_val.columns, 
                                 index=X_val.index)
X_val2_scaled_df = pd.DataFrame(X_val2_scaled, 
                                 columns=X_val2.columns, 
                                 index=X_val2.index)


# Let's assume you have data split into train/validation/test sets
# X_train, y_train, X_val, y_val, X_test, y_test
# and your data is properly scaled/encoded as needed.

# Convert your DataFrames to NumPy if you're working with pandas DataFrames.
# e.g., X_train_np = X_train_scaled_df.values

X_train_np = X_train_scaled_df.values
y_train_np = y_train.values
X_val_np   = X_val_scaled_df.values
y_val_np   = y_val.values
X_val2_np   = X_val2_scaled_df.values
y_val2_np   = y_val2.values


best_coefficients = []
best_test_score = 0
results = []
          
best_mask, best_acc = genetic_algorithm_feature_selection(
    X_train_np, y_train_np,
    X_val_np,   y_val_np,X_val2_np,   y_val2_np,
    pop_size=80,
    generations=800,
    mutation_rate=0.2,elitism=True,coefficients=[0.8,0,0,0.1,0]
)

feature_names = list(X_train_scaled_df.columns)

# Once you have the best_mask, you can evaluate on the test set:
selected_indices = [i for i, bit in enumerate(best_mask) if bit == 1]
model = DecisionTreeClassifier(max_depth=10)
#model = XGBClassifier(n_estimators=300,colsample_bytree= 0.8,gamma=0.1,learning_rate=0.1,max_depth=5,min_child_weight=1,subsample=0.9)
#model = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf = 20,max_features='sqrt')
model.fit(X_train_np[:, selected_indices], y_train_np)
y_pred_val = model.predict(X_val_np[:, selected_indices])
final_val_acc = accuracy_score(y_val_np, y_pred_val)
print("Best GA mask:", best_mask)
print("Validation score of best mask:", best_acc)
print("Validation accuracy of best mask:", final_val_acc)

selected_indices = [i for i, bit in enumerate(best_mask) if bit == 1]
best_feature_names = [feature_names[i] for i in selected_indices]
print("Selected features:", best_feature_names)

# Validation score of best mask: 0.5914035575319623
# Validation accuracy of best mask: 0.6178988326848249
# Selected features: ['date_after_0724', 'PM_min_to_open_ratio_class', 'PM_range_to_close_ratio_class', 'dayOpen_to_prevDayOpen_ratio_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'AH_max_1dayago_to_Close_class', 'Close_to_prevDayOpen_class', 'PM_min_to_prevDayOpen_ratio_class', 'close_to_High10_class', 'Close_class']

# Validation score of best mask: 0.5842094703049759
# Validation accuracy of best mask: 0.6211878009630819
# Selected features: ['return_2d_class', 'open_to_prev_close_class', 'PM_min_to_open_ratio_class', 'Close_to_open_ratio_class', 'dayOpen_to_prevDayClose_class', 'dayOpen_to_prevDayOpen_ratio_class', 'prev2DayClose_to_prevDayClose_ratio_class', 'AH_max_1dayago_vs_PM_max_class', 'AH_max_1dayago_vs_prevDayClose_class', 'Close_to_prevDayClose_class', 'Close_to_prevDayOpen_class', 'PM_min_to_prevDayOpen_ratio_class', 'PM_max_to_Close_ratio_class', 'PM_min_to_Close_ratio_class', 'Close_to_prevDayLow_class', 'ema_ratio1_class', 'ema_ratio2_class', 'close_to_Low10_class', 'vol_5_class'] 