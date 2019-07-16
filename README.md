# Kickstarter Model

Binary Classification model that predicts if a Kickstarter project will be successfully funded or not.

## Introduction

Kickstarter is a community of more than 10 million people comprising of creative, tech enthusiasts who help in bringing creative project to life.

Until now, more than $3 billion dollars have been contributed by the members in fueling creative projects. The projects can be literally anything – a device, a game, an app, a film etc.

Kickstarter works on all or nothing basis i.e if a project doesn’t meet its goal, the project owner gets nothing.
For example: if a projects’s goal is $5000. Even if it gets funded till $4999, the project won’t be a success.

## Goal

The goal of this project is to build a classifier that predicts whether a project will be successfully funded or not.

## The data

The source data consist of 25 input features and one binary target variable with values either ‘successful’ or ‘failed’. The type and format of the 25 input features is very diverse, including integer, float, boolean, categorical, text and json variables.

## The model

The Kickstarter Binary Classifier uses a Random Forest Classifier to classify projects as successful or unsuccessful. Data preprocessing and cleansing is crucial in this model to extract relevant features of the data. There are multiple categorical variables that are encoded using One Hot Encoding.

## Structure

model.py: this file contains the KickstarterModel class, which is the class of the Kickstarter Binary Classifier Model. The KickstarterModel class main methods are:
	- preprocess_training_data: preprocessing steps for training data
	- Preprocess_unseen_data: preprocessing steps for testing data
	- Preprocess_common: preprocessing steps which are common for testing and training data
	- fit: method to train the model
	- predict: method to generate predictions


run.py: python script to download the data, train the model and test it. Once trained, the model is serialised in a pickle file.

## Setup

Download the python files of the project and execute ‘python run.py setup’ to download the source data into the ‘data’ directory.

## Usage

python run.py train : Instanciates and trains a Kickstarter Binary Classification Model.

python run.py test : Generates predictions for the test data using a trained Kickstarter Binary Classification Model



