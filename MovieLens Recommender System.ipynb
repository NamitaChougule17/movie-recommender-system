{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "k-DodZFH9SDP"
   },
   "source": [
    "# Recommender Systems\n",
    "Implementing the Matrix Factorization algorithm."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "je5uChUl9SDS"
   },
   "source": [
    "# Part 1. Matrix Factorization\n",
    "\n",
    "For this part, we will:\n",
    "\n",
    "* load and process the MovieLens 1M dataset,\n",
    "* build a matrix factorization model,\n",
    "* evaluate the model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dtidMfaS9SDS"
   },
   "source": [
    "To start out, we need to prepare the data. We will use the MovieLens 1M data from https://grouplens.org/datasets/movielens/1m/ in this homework."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "executionInfo": {
     "elapsed": 5996,
     "status": "ok",
     "timestamp": 1732815259782,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "xmvxfcBd9SDT",
    "outputId": "6700f5c7-4951-427b-9f78-f225bb72223d"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.google.colaboratory.intrinsic+json": {
       "type": "dataframe",
       "variable_name": "data_df"
      },
      "text/html": [
       "\n",
       "  <div id=\"df-f3c7e423-ee8c-4217-8ae2-433bbeab3b6a\" class=\"colab-df-container\">\n",
       "    <div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>UserID</th>\n",
       "      <th>MovieID</th>\n",
       "      <th>Rating</th>\n",
       "      <th>Timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>1193</td>\n",
       "      <td>5</td>\n",
       "      <td>978300760</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>661</td>\n",
       "      <td>3</td>\n",
       "      <td>978302109</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>914</td>\n",
       "      <td>3</td>\n",
       "      <td>978301968</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>3408</td>\n",
       "      <td>4</td>\n",
       "      <td>978300275</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>2355</td>\n",
       "      <td>5</td>\n",
       "      <td>978824291</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>\n",
       "    <div class=\"colab-df-buttons\">\n",
       "\n",
       "  <div class=\"colab-df-container\">\n",
       "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f3c7e423-ee8c-4217-8ae2-433bbeab3b6a')\"\n",
       "            title=\"Convert this dataframe to an interactive table.\"\n",
       "            style=\"display:none;\">\n",
       "\n",
       "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
       "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
       "  </svg>\n",
       "    </button>\n",
       "\n",
       "  <style>\n",
       "    .colab-df-container {\n",
       "      display:flex;\n",
       "      gap: 12px;\n",
       "    }\n",
       "\n",
       "    .colab-df-convert {\n",
       "      background-color: #E8F0FE;\n",
       "      border: none;\n",
       "      border-radius: 50%;\n",
       "      cursor: pointer;\n",
       "      display: none;\n",
       "      fill: #1967D2;\n",
       "      height: 32px;\n",
       "      padding: 0 0 0 0;\n",
       "      width: 32px;\n",
       "    }\n",
       "\n",
       "    .colab-df-convert:hover {\n",
       "      background-color: #E2EBFA;\n",
       "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
       "      fill: #174EA6;\n",
       "    }\n",
       "\n",
       "    .colab-df-buttons div {\n",
       "      margin-bottom: 4px;\n",
       "    }\n",
       "\n",
       "    [theme=dark] .colab-df-convert {\n",
       "      background-color: #3B4455;\n",
       "      fill: #D2E3FC;\n",
       "    }\n",
       "\n",
       "    [theme=dark] .colab-df-convert:hover {\n",
       "      background-color: #434B5C;\n",
       "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
       "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
       "      fill: #FFFFFF;\n",
       "    }\n",
       "  </style>\n",
       "\n",
       "    <script>\n",
       "      const buttonEl =\n",
       "        document.querySelector('#df-f3c7e423-ee8c-4217-8ae2-433bbeab3b6a button.colab-df-convert');\n",
       "      buttonEl.style.display =\n",
       "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
       "\n",
       "      async function convertToInteractive(key) {\n",
       "        const element = document.querySelector('#df-f3c7e423-ee8c-4217-8ae2-433bbeab3b6a');\n",
       "        const dataTable =\n",
       "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
       "                                                    [key], {});\n",
       "        if (!dataTable) return;\n",
       "\n",
       "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
       "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
       "          + ' to learn more about interactive tables.';\n",
       "        element.innerHTML = '';\n",
       "        dataTable['output_type'] = 'display_data';\n",
       "        await google.colab.output.renderOutput(dataTable, element);\n",
       "        const docLink = document.createElement('div');\n",
       "        docLink.innerHTML = docLinkHtml;\n",
       "        element.appendChild(docLink);\n",
       "      }\n",
       "    </script>\n",
       "  </div>\n",
       "\n",
       "\n",
       "<div id=\"df-cfa6b111-6efa-45b9-8fa9-0bd5e7acfaaf\">\n",
       "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-cfa6b111-6efa-45b9-8fa9-0bd5e7acfaaf')\"\n",
       "            title=\"Suggest charts\"\n",
       "            style=\"display:none;\">\n",
       "\n",
       "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
       "     width=\"24px\">\n",
       "    <g>\n",
       "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
       "    </g>\n",
       "</svg>\n",
       "  </button>\n",
       "\n",
       "<style>\n",
       "  .colab-df-quickchart {\n",
       "      --bg-color: #E8F0FE;\n",
       "      --fill-color: #1967D2;\n",
       "      --hover-bg-color: #E2EBFA;\n",
       "      --hover-fill-color: #174EA6;\n",
       "      --disabled-fill-color: #AAA;\n",
       "      --disabled-bg-color: #DDD;\n",
       "  }\n",
       "\n",
       "  [theme=dark] .colab-df-quickchart {\n",
       "      --bg-color: #3B4455;\n",
       "      --fill-color: #D2E3FC;\n",
       "      --hover-bg-color: #434B5C;\n",
       "      --hover-fill-color: #FFFFFF;\n",
       "      --disabled-bg-color: #3B4455;\n",
       "      --disabled-fill-color: #666;\n",
       "  }\n",
       "\n",
       "  .colab-df-quickchart {\n",
       "    background-color: var(--bg-color);\n",
       "    border: none;\n",
       "    border-radius: 50%;\n",
       "    cursor: pointer;\n",
       "    display: none;\n",
       "    fill: var(--fill-color);\n",
       "    height: 32px;\n",
       "    padding: 0;\n",
       "    width: 32px;\n",
       "  }\n",
       "\n",
       "  .colab-df-quickchart:hover {\n",
       "    background-color: var(--hover-bg-color);\n",
       "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
       "    fill: var(--button-hover-fill-color);\n",
       "  }\n",
       "\n",
       "  .colab-df-quickchart-complete:disabled,\n",
       "  .colab-df-quickchart-complete:disabled:hover {\n",
       "    background-color: var(--disabled-bg-color);\n",
       "    fill: var(--disabled-fill-color);\n",
       "    box-shadow: none;\n",
       "  }\n",
       "\n",
       "  .colab-df-spinner {\n",
       "    border: 2px solid var(--fill-color);\n",
       "    border-color: transparent;\n",
       "    border-bottom-color: var(--fill-color);\n",
       "    animation:\n",
       "      spin 1s steps(1) infinite;\n",
       "  }\n",
       "\n",
       "  @keyframes spin {\n",
       "    0% {\n",
       "      border-color: transparent;\n",
       "      border-bottom-color: var(--fill-color);\n",
       "      border-left-color: var(--fill-color);\n",
       "    }\n",
       "    20% {\n",
       "      border-color: transparent;\n",
       "      border-left-color: var(--fill-color);\n",
       "      border-top-color: var(--fill-color);\n",
       "    }\n",
       "    30% {\n",
       "      border-color: transparent;\n",
       "      border-left-color: var(--fill-color);\n",
       "      border-top-color: var(--fill-color);\n",
       "      border-right-color: var(--fill-color);\n",
       "    }\n",
       "    40% {\n",
       "      border-color: transparent;\n",
       "      border-right-color: var(--fill-color);\n",
       "      border-top-color: var(--fill-color);\n",
       "    }\n",
       "    60% {\n",
       "      border-color: transparent;\n",
       "      border-right-color: var(--fill-color);\n",
       "    }\n",
       "    80% {\n",
       "      border-color: transparent;\n",
       "      border-right-color: var(--fill-color);\n",
       "      border-bottom-color: var(--fill-color);\n",
       "    }\n",
       "    90% {\n",
       "      border-color: transparent;\n",
       "      border-bottom-color: var(--fill-color);\n",
       "    }\n",
       "  }\n",
       "</style>\n",
       "\n",
       "  <script>\n",
       "    async function quickchart(key) {\n",
       "      const quickchartButtonEl =\n",
       "        document.querySelector('#' + key + ' button');\n",
       "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
       "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
       "      try {\n",
       "        const charts = await google.colab.kernel.invokeFunction(\n",
       "            'suggestCharts', [key], {});\n",
       "      } catch (error) {\n",
       "        console.error('Error during call to suggestCharts:', error);\n",
       "      }\n",
       "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
       "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
       "    }\n",
       "    (() => {\n",
       "      let quickchartButtonEl =\n",
       "        document.querySelector('#df-cfa6b111-6efa-45b9-8fa9-0bd5e7acfaaf button');\n",
       "      quickchartButtonEl.style.display =\n",
       "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
       "    })();\n",
       "  </script>\n",
       "</div>\n",
       "\n",
       "    </div>\n",
       "  </div>\n"
      ],
      "text/plain": [
       "   UserID  MovieID  Rating  Timestamp\n",
       "0       1     1193       5  978300760\n",
       "1       1      661       3  978302109\n",
       "2       1      914       3  978301968\n",
       "3       1     3408       4  978300275\n",
       "4       1     2355       5  978824291"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "data_df = pd.read_csv('./ratings.dat', sep='::',\n",
    "                      names=[\"UserID\", \"MovieID\", \"Rating\", \"Timestamp\"],\n",
    "                      engine='python')\n",
    "data_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 260,
     "status": "ok",
     "timestamp": 1732815265068,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "gF8_u3Yj9SDU",
    "outputId": "97b48450-faf1-4038-a085-ef36a7b444fb"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6040, 3706, 1000209)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Counting unique users, unique movies, and total ratings\n",
    "unique_users = data_df['UserID'].nunique()\n",
    "unique_movies = data_df['MovieID'].nunique()\n",
    "total_ratings = data_df['Rating'].count()\n",
    "\n",
    "# Printing the results\n",
    "unique_users, unique_movies, total_ratings"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JAT_-YZw9SDV"
   },
   "source": [
    "Because in Python, the index for a list starts from 0, it is more convenient if we have the ids of users and movies start from 0 as well. Moreover, we also need to make sure the UserID and MovieID are consecutive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 206
    },
    "executionInfo": {
     "elapsed": 84749,
     "status": "ok",
     "timestamp": 1732815353103,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "tHjgPm3V9SDV",
    "outputId": "92ab8ffe-855b-4af2-912a-7ef989c68e82"
   },
   "outputs": [
    {
     "data": {
      "application/vnd.google.colaboratory.intrinsic+json": {
       "type": "dataframe",
       "variable_name": "data_df"
      },
      "text/html": [
       "\n",
       "  <div id=\"df-853f701a-e3d1-440f-b237-89826280e456\" class=\"colab-df-container\">\n",
       "    <div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>UserID</th>\n",
       "      <th>MovieID</th>\n",
       "      <th>Rating</th>\n",
       "      <th>Timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5</td>\n",
       "      <td>978300760</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>978302109</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>978301968</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>978300275</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>978824291</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>\n",
       "    <div class=\"colab-df-buttons\">\n",
       "\n",
       "  <div class=\"colab-df-container\">\n",
       "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-853f701a-e3d1-440f-b237-89826280e456')\"\n",
       "            title=\"Convert this dataframe to an interactive table.\"\n",
       "            style=\"display:none;\">\n",
       "\n",
       "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
       "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
       "  </svg>\n",
       "    </button>\n",
       "\n",
       "  <style>\n",
       "    .colab-df-container {\n",
       "      display:flex;\n",
       "      gap: 12px;\n",
       "    }\n",
       "\n",
       "    .colab-df-convert {\n",
       "      background-color: #E8F0FE;\n",
       "      border: none;\n",
       "      border-radius: 50%;\n",
       "      cursor: pointer;\n",
       "      display: none;\n",
       "      fill: #1967D2;\n",
       "      height: 32px;\n",
       "      padding: 0 0 0 0;\n",
       "      width: 32px;\n",
       "    }\n",
       "\n",
       "    .colab-df-convert:hover {\n",
       "      background-color: #E2EBFA;\n",
       "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
       "      fill: #174EA6;\n",
       "    }\n",
       "\n",
       "    .colab-df-buttons div {\n",
       "      margin-bottom: 4px;\n",
       "    }\n",
       "\n",
       "    [theme=dark] .colab-df-convert {\n",
       "      background-color: #3B4455;\n",
       "      fill: #D2E3FC;\n",
       "    }\n",
       "\n",
       "    [theme=dark] .colab-df-convert:hover {\n",
       "      background-color: #434B5C;\n",
       "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
       "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
       "      fill: #FFFFFF;\n",
       "    }\n",
       "  </style>\n",
       "\n",
       "    <script>\n",
       "      const buttonEl =\n",
       "        document.querySelector('#df-853f701a-e3d1-440f-b237-89826280e456 button.colab-df-convert');\n",
       "      buttonEl.style.display =\n",
       "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
       "\n",
       "      async function convertToInteractive(key) {\n",
       "        const element = document.querySelector('#df-853f701a-e3d1-440f-b237-89826280e456');\n",
       "        const dataTable =\n",
       "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
       "                                                    [key], {});\n",
       "        if (!dataTable) return;\n",
       "\n",
       "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
       "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
       "          + ' to learn more about interactive tables.';\n",
       "        element.innerHTML = '';\n",
       "        dataTable['output_type'] = 'display_data';\n",
       "        await google.colab.output.renderOutput(dataTable, element);\n",
       "        const docLink = document.createElement('div');\n",
       "        docLink.innerHTML = docLinkHtml;\n",
       "        element.appendChild(docLink);\n",
       "      }\n",
       "    </script>\n",
       "  </div>\n",
       "\n",
       "\n",
       "<div id=\"df-95d42ed2-cb56-4b6d-9f21-1b1d0c4a92fb\">\n",
       "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-95d42ed2-cb56-4b6d-9f21-1b1d0c4a92fb')\"\n",
       "            title=\"Suggest charts\"\n",
       "            style=\"display:none;\">\n",
       "\n",
       "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
       "     width=\"24px\">\n",
       "    <g>\n",
       "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
       "    </g>\n",
       "</svg>\n",
       "  </button>\n",
       "\n",
       "<style>\n",
       "  .colab-df-quickchart {\n",
       "      --bg-color: #E8F0FE;\n",
       "      --fill-color: #1967D2;\n",
       "      --hover-bg-color: #E2EBFA;\n",
       "      --hover-fill-color: #174EA6;\n",
       "      --disabled-fill-color: #AAA;\n",
       "      --disabled-bg-color: #DDD;\n",
       "  }\n",
       "\n",
       "  [theme=dark] .colab-df-quickchart {\n",
       "      --bg-color: #3B4455;\n",
       "      --fill-color: #D2E3FC;\n",
       "      --hover-bg-color: #434B5C;\n",
       "      --hover-fill-color: #FFFFFF;\n",
       "      --disabled-bg-color: #3B4455;\n",
       "      --disabled-fill-color: #666;\n",
       "  }\n",
       "\n",
       "  .colab-df-quickchart {\n",
       "    background-color: var(--bg-color);\n",
       "    border: none;\n",
       "    border-radius: 50%;\n",
       "    cursor: pointer;\n",
       "    display: none;\n",
       "    fill: var(--fill-color);\n",
       "    height: 32px;\n",
       "    padding: 0;\n",
       "    width: 32px;\n",
       "  }\n",
       "\n",
       "  .colab-df-quickchart:hover {\n",
       "    background-color: var(--hover-bg-color);\n",
       "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
       "    fill: var(--button-hover-fill-color);\n",
       "  }\n",
       "\n",
       "  .colab-df-quickchart-complete:disabled,\n",
       "  .colab-df-quickchart-complete:disabled:hover {\n",
       "    background-color: var(--disabled-bg-color);\n",
       "    fill: var(--disabled-fill-color);\n",
       "    box-shadow: none;\n",
       "  }\n",
       "\n",
       "  .colab-df-spinner {\n",
       "    border: 2px solid var(--fill-color);\n",
       "    border-color: transparent;\n",
       "    border-bottom-color: var(--fill-color);\n",
       "    animation:\n",
       "      spin 1s steps(1) infinite;\n",
       "  }\n",
       "\n",
       "  @keyframes spin {\n",
       "    0% {\n",
       "      border-color: transparent;\n",
       "      border-bottom-color: var(--fill-color);\n",
       "      border-left-color: var(--fill-color);\n",
       "    }\n",
       "    20% {\n",
       "      border-color: transparent;\n",
       "      border-left-color: var(--fill-color);\n",
       "      border-top-color: var(--fill-color);\n",
       "    }\n",
       "    30% {\n",
       "      border-color: transparent;\n",
       "      border-left-color: var(--fill-color);\n",
       "      border-top-color: var(--fill-color);\n",
       "      border-right-color: var(--fill-color);\n",
       "    }\n",
       "    40% {\n",
       "      border-color: transparent;\n",
       "      border-right-color: var(--fill-color);\n",
       "      border-top-color: var(--fill-color);\n",
       "    }\n",
       "    60% {\n",
       "      border-color: transparent;\n",
       "      border-right-color: var(--fill-color);\n",
       "    }\n",
       "    80% {\n",
       "      border-color: transparent;\n",
       "      border-right-color: var(--fill-color);\n",
       "      border-bottom-color: var(--fill-color);\n",
       "    }\n",
       "    90% {\n",
       "      border-color: transparent;\n",
       "      border-bottom-color: var(--fill-color);\n",
       "    }\n",
       "  }\n",
       "</style>\n",
       "\n",
       "  <script>\n",
       "    async function quickchart(key) {\n",
       "      const quickchartButtonEl =\n",
       "        document.querySelector('#' + key + ' button');\n",
       "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
       "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
       "      try {\n",
       "        const charts = await google.colab.kernel.invokeFunction(\n",
       "            'suggestCharts', [key], {});\n",
       "      } catch (error) {\n",
       "        console.error('Error during call to suggestCharts:', error);\n",
       "      }\n",
       "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
       "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
       "    }\n",
       "    (() => {\n",
       "      let quickchartButtonEl =\n",
       "        document.querySelector('#df-95d42ed2-cb56-4b6d-9f21-1b1d0c4a92fb button');\n",
       "      quickchartButtonEl.style.display =\n",
       "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
       "    })();\n",
       "  </script>\n",
       "</div>\n",
       "\n",
       "    </div>\n",
       "  </div>\n"
      ],
      "text/plain": [
       "   UserID  MovieID  Rating  Timestamp\n",
       "0       0        0       5  978300760\n",
       "1       0        1       3  978302109\n",
       "2       0        2       3  978301968\n",
       "3       0        3       4  978300275\n",
       "4       0        4       5  978824291"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# First, generate dictionaries for mapping old id to new id for users and movies\n",
    "unique_MovieID = data_df['MovieID'].unique()\n",
    "unique_UserID = data_df['UserID'].unique()\n",
    "j = 0\n",
    "user_old2new_id_dict = dict()\n",
    "for u in unique_UserID:\n",
    "    user_old2new_id_dict[u] = j\n",
    "    j += 1\n",
    "j = 0\n",
    "movie_old2new_id_dict = dict()\n",
    "for i in unique_MovieID:\n",
    "    movie_old2new_id_dict[i] = j\n",
    "    j += 1\n",
    "\n",
    "# Then, use the generated dictionaries to reindex UserID and MovieID in the data_df\n",
    "for j in range(len(data_df)):\n",
    "    data_df.at[j, 'UserID'] = user_old2new_id_dict[data_df.at[j, 'UserID']]\n",
    "    data_df.at[j, 'MovieID'] = movie_old2new_id_dict[data_df.at[j, 'MovieID']]\n",
    "\n",
    "data_df.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "upf-GND_9SDV"
   },
   "source": [
    "Now, you have got a ready-to-use dataset. The next step is to split it **randomly** into training and testing sets so that you can build your recommendation model based on the training set and evaluate it on the testing set. Here you need to split the data_df into two parts: a DataFrame **train_df** containing 70% user-movie-rating samples in data_df, and a DataFrame **test_df** containing 30% samples. train_df and test_df should have no overlap. In the next cell, write your code and print the numbers of samples in the generated train_df and test_df at last.\n",
    "\n",
    "**Note that here we just have training and testing sets without using a validation set for the sake of simplicity.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 150,
     "status": "ok",
     "timestamp": 1732815471073,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "6JGBl2M59SDV",
    "outputId": "1c8a408e-2ae8-4683-9a98-581c79aeba7c"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "num of train_df = 700146\n",
      "num of test_df = 300063\n"
     ]
    }
   ],
   "source": [
    "# generate train_df with 70% samples and test_df with 30% samples, and there should have no overlap between them.\n",
    "# Your Code Here...\n",
    "# Set random seed for reproducibility\n",
    "np.random.seed(42)\n",
    "\n",
    "# Shuffle the indices of the DataFrame\n",
    "shuffled_indices = np.random.permutation(len(data_df))\n",
    "\n",
    "# Split indices for training (70%) and testing (30%)\n",
    "train_size = int(len(data_df) * 0.7)\n",
    "train_indices = shuffled_indices[:train_size]\n",
    "test_indices = shuffled_indices[train_size:]\n",
    "\n",
    "# Use the indices to split the DataFrame\n",
    "train_df = data_df.iloc[train_indices].reset_index(drop=True)\n",
    "test_df = data_df.iloc[test_indices].reset_index(drop=True)\n",
    "# End of your code\n",
    "\n",
    "print('num of train_df = ' + str(len(train_df)))\n",
    "print('num of test_df = ' + str(len(test_df)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mq-7Wcwz9SDW"
   },
   "source": [
    "Last, we need to generate numpy array variables (i.e., matrix version of dataset) for both train_df and test_df for the ease of calculation in the next step. More specifically, we will generate two numpy array variables of size (#user by #movie) with each entry representing the user-movie rating. And if the user-movie rating is missing, then the corresponding entry is 0. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "executionInfo": {
     "elapsed": 946,
     "status": "ok",
     "timestamp": 1732815476408,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "ci6kU_Cq9SDW"
   },
   "outputs": [],
   "source": [
    "from scipy.sparse import coo_matrix\n",
    "\n",
    "num_user = len(data_df['UserID'].unique())\n",
    "num_movie = len(data_df['MovieID'].unique())\n",
    "\n",
    "train_mat = coo_matrix((train_df['Rating'].values, (train_df['UserID'].values, train_df['MovieID'].values)),\n",
    "                       shape=(num_user, num_movie)).toarray().astype(float)\n",
    "test_mat = coo_matrix((test_df['Rating'].values, (test_df['UserID'].values, test_df['MovieID'].values)),\n",
    "                      shape=(num_user, num_movie)).toarray().astype(float)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mTOQWA4C9SDW"
   },
   "source": [
    "With the processed data, let's implement the matrix factorization (MF for short) model. The MF model can be mathematically represented as:\n",
    "\n",
    "<center>$\\underset{\\mathbf{P},\\mathbf{Q}}{\\text{min}}\\,\\,L=\\sum_{(u,i)\\in\\mathcal{O}}(\\mathbf{P}_u\\cdot\\mathbf{Q}^\\top_i-r_{u,i})^2+\\lambda(\\lVert\\mathbf{P}\\rVert^2_{\\text{F}}+\\lVert\\mathbf{Q}\\rVert^2_{\\text{F}})$,</center>\n",
    "    \n",
    "where $\\mathbf{P}$ is the user latent factor matrix of size (#user, #latent); $\\mathbf{Q}$ is the movie latent factor matrix of size (#movie, #latent); $\\mathcal{O}$ is a user-movie pair set containing all user-movie pairs having ratings in train_mat; $r_{u,i}$ represents the rating for user u and movie i; $\\lambda(\\lVert\\mathbf{P}\\rVert^2_{\\text{F}}+\\lVert\\mathbf{Q}\\rVert^2_{\\text{F}})$ is the regularization term to overcome overfitting problem, $\\lambda$ is the regularization weight (a hyper-parameter manually set by developer, i.e., you), and $\\lVert\\mathbf{P}\\rVert^2_{\\text{F}}=\\sum_{x}\\sum_{y}(\\mathbf{P}_{x,y})^2$, $\\lVert\\mathbf{Q}\\rVert^2_{\\text{F}}=\\sum_{x}\\sum_{y}(\\mathbf{Q}_{x,y})^2$. Such an L function is called the **loss function** for the matrix factorization model. The goal of training an MF model is to find appropriate $\\mathbf{P}$ and $\\mathbf{Q}$ to minimize the loss L."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "k1XdS4ls9SDW"
   },
   "source": [
    "To implement such an MF, here we will write a Python class for the model. There are three functions in this MF class: init, train, and predict.\n",
    "\n",
    "* The 'init' function (**already provided**) is to initialize the variables the MF class needs, which takes 5 inputs: train_mat, test_mat, latent, lr, and reg. 'train_mat' and 'test_mat' are the corresponfing training and testing matrices we have. 'latent' represents the latent dimension we set for the MF model. 'lr' represents the learning rate, i.e., the update step in each optimization iteration, default is 0.01. 'reg' represents the regularization weight, i.e., the $\\lambda$ in the MF formulation.\n",
    "\n",
    "* The 'train' function (**partially provided and need to complete**) is to train the MF model given the training data train_mat. There is only one input to this function: an int variable 'epoch' to indicate how many epochs for training the model. The main body of this function should be a loop for 'epoch' iterations. In each iteration, following the algorithm to update the MF model:\n",
    "\n",
    "        1. Randomly shuffle training user-movie pairs  (i.e., user-movie pairs having ratings in train_mat)\n",
    "        2. Have an inner loop to iterate each user-movie pair:\n",
    "                a. given a user-movie pair (u,i), update the user latent factor and movie latent factor by gradient decsent:    \n",
    "<center>$\\mathbf{P}^\\prime_u=\\mathbf{P}_u-\\gamma [2(\\mathbf{P}_u\\cdot\\mathbf{Q}_i^\\top-r_{u,i})\\cdot\\mathbf{Q}_i+2\\lambda\\mathbf{P}_u]$</center>    \n",
    "<center>$\\mathbf{Q}^\\prime_i=\\mathbf{Q}_i-\\gamma [2(\\mathbf{P}_u\\cdot\\mathbf{Q}_i^\\top-r_{u,i})\\cdot\\mathbf{P}_u+2\\lambda\\mathbf{Q}_i]$</center>    \n",
    "<center>where $\\mathbf{P}_u$ and $\\mathbf{Q}_i$ are row vectors of size (1, #latent), $\\gamma$ is learning rate (default is 0.01), $\\lambda$ is regularization weight.</center>\n",
    "        \n",
    "        3. After iterating over all user-movie pairs, we have finished the training for the current epoch. Now calculate and print out the value of the loss function L after this epoch, and the RMSE on test_mat by the current MF model. Then append them to lists to keep a record of them.\n",
    "The train function needs to return two lists: 'epoch_loss_list' recording the loss after each training epoch, and 'epoch_test_RMSE_list' recording the RMSE on test_mat after each training epoch. The calculation of RMSE is formulated as:\n",
    "<center>$RMSE=\\sqrt{\\frac{1}{|\\mathcal{O}_{test}|}\\sum_{(u,i)\\in\\mathcal{O}_{test}}(\\mathbf{P}_u\\cdot\\mathbf{Q}^\\top_i-r_{u,i})^2}$</center>\n",
    "<center>where $\\mathcal{O}_{test}$ is a user-movie pair set containing all user-movie pairs having ratings in test_mat, and $|\\mathcal{O}_{test}|$ represents the total number of user-movie pairs in test_mat.</center>\n",
    "\n",
    "* The 'predict' function (**already provided**) is to calculate the prediction_mat by the learned $\\mathbf{P}$ and $\\mathbf{Q}$.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "executionInfo": {
     "elapsed": 263,
     "status": "ok",
     "timestamp": 1732815481957,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "vrI8jwLf9SDW"
   },
   "outputs": [],
   "source": [
    "class MF:\n",
    "    def __init__(self, train_mat, test_mat, latent=5, lr=0.01, reg=0.01):\n",
    "        self.train_mat = train_mat  # the training rating matrix of size (#user, #movie)\n",
    "        self.test_mat = test_mat  # the training rating matrix of size (#user, #movie)\n",
    "\n",
    "        self.latent = latent  # the latent dimension\n",
    "        self.lr = lr  # learning rate\n",
    "        self.reg = reg  # regularization weight, i.e., the lambda in the objective function\n",
    "\n",
    "        self.num_user, self.num_movie = train_mat.shape\n",
    "\n",
    "        # get the user-movie pairs having ratings in train_mat\n",
    "        # self.train_user stores user ids, self.train_movie stores movie ids.\n",
    "        self.train_user, self.train_movie = self.train_mat.nonzero()\n",
    "\n",
    "        self.num_train = len(self.train_user)  # the number of user-movie pairs having ratings in train_mat\n",
    "\n",
    "        self.train_indicator_mat = 1.0 * (train_mat > 0)  # binary matrix to indicate whether s user-movie pair has rating or not in train_mat\n",
    "        self.test_indicator_mat = 1.0 * (test_mat > 0)  # binary matrix to indicate whether s user-movie pair has rating or not in test_mat\n",
    "\n",
    "        self.P = np.random.random((self.num_user, self.latent))  # latent factors for users, size (#user, self.latent), randomly initialized\n",
    "        self.Q = np.random.random((self.num_movie, self.latent))  # latent factors for users, size (#movie, self.latent), randomly initialized\n",
    "\n",
    "    def train(self, epoch=20, verbose=True):\n",
    "        \"\"\"\n",
    "        Goal: Write your code to train your matrix factorization model for epoch iterations in this function\n",
    "        Input: epoch -- the number of training epoch\n",
    "        Output: epoch_loss_list -- a list recording the training loss for each epoch\n",
    "                epoch_test_RMSE_list -- a list recording the testing RMSE after each training epoch\n",
    "        \"\"\"\n",
    "        epoch_loss_list = []\n",
    "        epoch_test_RMSE_list = []\n",
    "        for ep in range(epoch):\n",
    "            \"\"\"\n",
    "            Write your code here to implement the training process for one epoch,\n",
    "            and at the end of each epoch, print out the epoch number, the training loss after this epoch,\n",
    "            and the test RMSE after this epoch\n",
    "            \"\"\"\n",
    "            # start of your code\n",
    "            indices = np.random.permutation(self.num_train)\n",
    "            train_user_shuffled = self.train_user[indices]\n",
    "            train_movie_shuffled = self.train_movie[indices]\n",
    "\n",
    "            # Loop through shuffled user-movie pairs and update latent factors\n",
    "            for u, i in zip(train_user_shuffled, train_movie_shuffled):\n",
    "                rating = self.train_mat[u, i]\n",
    "                pred_rating = np.dot(self.P[u], self.Q[i].T)\n",
    "                error = rating - pred_rating\n",
    "\n",
    "                # Update user and movie latent factors using gradient descent\n",
    "                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])\n",
    "                self.Q[i] += self.lr * (error * self.P[u] - self.reg * self.Q[i])\n",
    "\n",
    "            # Compute training loss\n",
    "            pred_train_mat = np.matmul(self.P, self.Q.T)\n",
    "            train_loss = np.sum(((self.train_mat - pred_train_mat) * self.train_indicator_mat) ** 2) + \\\n",
    "                         self.reg * (np.sum(self.P ** 2) + np.sum(self.Q ** 2))\n",
    "            epoch_loss = train_loss / np.sum(self.train_indicator_mat)\n",
    "\n",
    "            # Compute test RMSE\n",
    "            pred_test_mat = np.matmul(self.P, self.Q.T)\n",
    "            test_rmse = (np.sum(((pred_test_mat - self.test_mat) * self.test_indicator_mat) ** 2) /\n",
    "                         np.sum(self.test_indicator_mat)) ** 0.5\n",
    "\n",
    "            # End of your code for this function\n",
    "\n",
    "            epoch_loss_list.append(epoch_loss)\n",
    "            epoch_test_RMSE_list.append(test_rmse)\n",
    "            if verbose:\n",
    "                print('Epoch={0}, Training Loss={1}, Testing RMSE={2}'.format(ep + 1, epoch_loss, test_rmse))\n",
    "\n",
    "        return epoch_loss_list, epoch_test_RMSE_list\n",
    "\n",
    "\n",
    "    def predict(self):\n",
    "        prediction_mat = np.matmul(self.P, self.Q.T)\n",
    "        return prediction_mat"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WFEDmOIJ9SDX"
   },
   "source": [
    "Now, let's train an MF model based on your implementation. The code is provided, you just need to excute the next cell. The expectations are:\n",
    "\n",
    "* first, the code can be successfully excuted without error;\n",
    "* second, the training loss and RMSE on **test_mat** of each training epoch should be printed out for all 20 epochs;\n",
    "* last, the best RMSE on **test_mat** should be <0.92.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 276776,
     "status": "ok",
     "timestamp": 1732815762978,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "c-UMpEs-9SDX",
    "outputId": "47f7dd34-0e65-425b-cbf1-e99925c459e8"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch=1, Training Loss=0.8771173076259403, Testing RMSE=0.9527735527214727\n",
      "Epoch=2, Training Loss=0.8404894562094769, Testing RMSE=0.9354959115096152\n",
      "Epoch=3, Training Loss=0.8301586276703067, Testing RMSE=0.9317693971276416\n",
      "Epoch=4, Training Loss=0.820443877993414, Testing RMSE=0.9291760779046014\n",
      "Epoch=5, Training Loss=0.8106226334513082, Testing RMSE=0.9269482471411353\n",
      "Epoch=6, Training Loss=0.7953073225002218, Testing RMSE=0.9221998658506183\n",
      "Epoch=7, Training Loss=0.7800130689687022, Testing RMSE=0.9170609802181863\n",
      "Epoch=8, Training Loss=0.7677995705465454, Testing RMSE=0.9120517986590219\n",
      "Epoch=9, Training Loss=0.7556188091597315, Testing RMSE=0.9074839714932564\n",
      "Epoch=10, Training Loss=0.747104769848709, Testing RMSE=0.9053649652383712\n",
      "Epoch=11, Training Loss=0.7395779085611831, Testing RMSE=0.9039088091153309\n",
      "Epoch=12, Training Loss=0.7324019955938464, Testing RMSE=0.9016873393699\n",
      "Epoch=13, Training Loss=0.7278831056981634, Testing RMSE=0.9012473688448605\n",
      "Epoch=14, Training Loss=0.7230874669408869, Testing RMSE=0.8996007823234126\n",
      "Epoch=15, Training Loss=0.7186364655530398, Testing RMSE=0.8990063584271246\n",
      "Epoch=16, Training Loss=0.7145845870142182, Testing RMSE=0.8982841784415414\n",
      "Epoch=17, Training Loss=0.7110615174261511, Testing RMSE=0.8961769511561497\n",
      "Epoch=18, Training Loss=0.7082292631856293, Testing RMSE=0.89580335327121\n",
      "Epoch=19, Training Loss=0.7035804319584511, Testing RMSE=0.8944209346927352\n",
      "Epoch=20, Training Loss=0.7023096942033618, Testing RMSE=0.8944388540571917\n"
     ]
    }
   ],
   "source": [
    "mf = MF(train_mat, test_mat, latent=5, lr=0.01, reg=0.001)\n",
    "epoch_loss_list, epoch_test_RMSE_list = mf.train(epoch=20)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "umZuxCEY9SDX"
   },
   "source": [
    "# Part 2: Tune Hyper-parameters in Matrix Factorization (30 points)\n",
    "\n",
    "In Part 1, we train an MF model with latent dimension set as 5, regularization weight as 0.001, training epoch as 20. However, it is not clear whether these are good choices or not. Hence, in this part, we will tune these hyper-parameters to train an effective model.\n",
    "\n",
    "A most straightforward but powerful method is to grid search each hyper-parameter and find the best one based on the RMSE on test_mat. In this part, we will do the grid search for train epoch, latent dimension, and regularization weight."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "caMnitav9SDX"
   },
   "source": [
    "### Tune training epoch\n",
    "\n",
    "For training epoch, we only need to run the experiment of MF for one time, and record the test RMSE for each epoch and find the epoch that produces the best test RMSE. To visually show the change of test RMSE corresponding to the training epoch, we can plot the test RMSE for each epoch in a figure as shown in the next cell."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 407
    },
    "executionInfo": {
     "elapsed": 506,
     "status": "ok",
     "timestamp": 1732817599770,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "Hgvtp7JF9SDX",
    "outputId": "22288a34-6efc-413b-c817-bc9be1147d29"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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YUFERE3K/v37IQgCsrOzpY5CRNSssLAgIiIiIqJaY2FBRERERES1xsKCiIjqlF6vx5IlS9C6dWuYmprCz88Pv/76K4C/b1Pavn07unTpArVajV69euHs2bMV9vHbb7+hU6dOUKlU8PT0xNKlSys8r9Fo8Oabb8LNzQ0qlQpt2rTBN998U6HPyZMn0a1bN5iZmaFPnz64ePFi/Z44EVEzx8KCiIjq1JIlS/D9999jzZo1OHfuHGbPno2JEyciMjLS0OeNN97A0qVLcfz4cdjb22PkyJHQarUAygqCcePG4emnn8aZM2ewcOFCvPvuu/juu+8M20+aNAk//PADPvvsM5w/fx5r166FhYVFhRzvvPMOli5dihMnTsDExATPP/+8Uc6fiKi5EkRRFKUOQURETYNGo4GtrS327NmD3r17G9pffPFFFBYW4qWXXsKgQYPw448/Yvz48QCArKwstGrVCt999x3GjRuHZ599FhkZGdi9e7dh+zlz5mD79u04d+4cLl26hPbt2yM8PByBgYGVMuzfvx+DBg3Cnj17MGTIEADAjh078Nhjj6GoqAhqtbqeXwUiouaJVyyIiKjOXL58GYWFhRg6dCgsLCwMX99//z2uXLli6PfPosPW1hbt27fH+fPnAQDnz59H3759K+y3b9++iI+Ph06nQ2xsLORyOQYMGHDfLF26dDH87OzsDABIT0+v9TkSEVHVTKQOQERETUd+fj4AYPv27XB1da3wnEqlqlBcPCxTU9Nq9VMoFIafBUEAUDb+g4iI6gevWBARUZ3p2LEjVCoVEhMT0aZNmwpfbm5uhn5Hjx41/Hz79m1cunQJHTp0AAB06NABhw8frrDfw4cPo127dpDL5ejcuTP0en2FMRtERCQ9XrEgIqI6Y2lpiddffx2zZ8+GXq9Hv379kJOTg8OHD8PKygoeHh4AgMWLF6Nly5ZwdHTEO++8Azs7O4wePRoA8Nprr6F79+547733MH78eERFReHzzz/HF198AQDw9PTE5MmT8fzzz+Ozzz6Dn58frl+/jvT0dIwbN06qUyciavZYWBARUZ167733YG9vjyVLliAhIQE2Njbo2rUr3n77bcOtSB988AFeffVVxMfHw9/fH1u3boVSqQQAdO3aFT///DPmz5+P9957D87Ozli8eDGmTJliOMbq1avx9ttv4z//+Q8yMzPh7u6Ot99+W4rTJSKiOzgrFBERGU35jE23b9+GjY2N1HGIiKgOcYwFERERERHVGgsLIiIiIiKqNd4KRUREREREtcYrFkREREREVGssLIiIiIiIqNZYWBARERERUa2xsCAiIiIiolpjYUFERERERLXGwoKIiIiIiGqNhQUREREREdUaCwsiIiIiIqo1FhZERERERFRr/w+ZkdN2onwNfAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 800x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))\n",
    "ax.plot(np.arange(len(epoch_test_RMSE_list)) + 1, epoch_test_RMSE_list, marker='.', linewidth=1.5, markersize=8)\n",
    "ax.set_xticks(np.arange(len(epoch_test_RMSE_list)) + 1)\n",
    "ax.set_ylabel('test RMSE')\n",
    "ax.set_xlabel('epoch')\n",
    "ax.set_title('Tune training epoch')\n",
    "ax.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vNXBqQDe9SDY"
   },
   "source": [
    "### Tune latent dimension\n",
    "\n",
    "By this figure, you can find the best epoch for your MF model. Similarly, you can plot how the test RMSE changes when you set different latent dimensions. For this, you need to run the code to initialize and train the MF model for multiple times with different settings of 'latent'. Please run the MF model with 'latent' as {1,3,5,7,9}, and plot the corresponding test RMSE for these five different latent dimensions in the next cell.\n",
    "\n",
    "For these five runs of experiments, record the test RMSE after Ep training epochs -- Ep is the best epoch you find by the 'Tune training epoch' plot. And here, fix regularization weight as 0.001.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 407
    },
    "executionInfo": {
     "elapsed": 1062301,
     "status": "ok",
     "timestamp": 1732818669387,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "a4QudzXX9SDY",
    "outputId": "82378bb7-a48c-44c7-fcf0-e1b98c0dd538"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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R47EqlWrcPv2bdTX12P79u04ceIEcnJyxB6eRiyF6sRYCkXtrba2FllZWSgtLcWuXbvw1Vdf4ciRIwwX1K5KSkrg4uKC6OholuFRuwoICICRkRF+/PFHsYdC3URaWhpee+01HD16FPr6+nj88cfRv39/JCcn49KlS2IPrwnesSAirRkZGaFv377w8fFBVFQUvLy8sG7dOrGHRd2MlZUV+vfvj+vXr4s9FOpGbty4gYMHD2LevHliD4W6EQ8PDxw5cgQVFRW4efMmTp06BYVCAXd3d7GHphGDBRE9MqVSCblcLvYwqJupqKhAWloa7O3txR4KdSObN29G79691RbRErWXHj16wN7eHnfu3EFcXBwmT54s9pA0MhB7AKSbiooKtd/SZWRkICUlBTY2NnB2dhZxZNTVhYeHY+LEiXB2dkZ5eTm++eYbHD58GHFxcWIPjbq4xYsXY9KkSXBxccHt27exYsUK6Ovr449//KPYQ6NuQqlUYvPmzZg9ezYMDPijE7WfuLg4CIKAAQMG4Pr163j33XcxcOBABAcHiz00jfhfRyeTlJSEsWPHql6HhYUBAGbPno0tW7aINCrqDvLz8zFr1izk5OTA0tISw4YNQ1xcHMaPHy/20KiLu3XrFv74xz+iqKgItra2ePrpp3Hy5EnY2tqKPTTqJg4ePIisrCy89tprYg+FupnS0lKEh4fj1q1bsLGxwfTp0/HRRx/B0NBQ7KFpxMXbRERERETUYlxjQURERERELcZgQURERERELcZgQURERERELcZgQURERERELcZgQURERERELcZgQURERERELcZgQURERERELcZgQURERERELcZgQUTURYwZMwZvvfWW2MNosTlz5mDKlCmq153luiQSCfbu3Sv2MIiIRGMg9gCIiEgchw8fxtixY3Hnzh1YWVm16rlXrlyJvXv3IiUlpcXn2r17NwwNDVs+qDaWk5MDa2trsYdBRCQaBgsiIurQbGxsxB6CVmQymdhDICISFUuhiIi6qJiYGPj6+sLc3BwymQwvv/wy8vPzAQCZmZkYO3YsAMDa2hoSiQRz5swBACiVSkRFRcHNzQ0mJibw8vLCrl27VOc9fPgwJBIJEhIS4OvrC1NTUzz55JO4cuUKAGDLli2IjIzEb7/9BolEAolEgi1btmgcY319PcLCwmBlZYWePXtiyZIlEARBrc39pVCurq748MMPMWvWLJiZmcHFxQU//PADCgoKMHnyZJiZmWHYsGFISkpSO8+xY8fwzDPPwMTEBE5OTnjzzTdRWVmpdt6//e1veO2112Bubg5nZ2d8+eWXqs/X1tYiNDQU9vb2kEqlcHFxQVRUlOrz95dCnT9/Hs8++yxMTEzQs2dPzJ8/HxUVFarPN5Z8/f3vf4e9vT169uyJRYsWQaFQNPdXSkTUoTFYEBF1UQqFAqtWrcJvv/2GvXv3IjMzUxUenJyc8P333wMArly5gpycHKxbtw4AEBUVhW3btmHjxo24ePEi3n77bbzyyis4cuSI2vnfe+89rFmzBklJSTAwMMBrr70GAJg5cybeeecdDB48GDk5OcjJycHMmTM1jnHNmjXYsmULNm3ahGPHjqG4uBh79ux56LV9+umneOqpp3D27Fk899xzePXVVzFr1iy88sorOHPmDDw8PDBr1ixVSElLS0NgYCCmT5+Oc+fOYceOHTh27BhCQ0ObjMfX1xdnz57Fn//8ZyxcuFAVmP7xj3/ghx9+wHfffYcrV67g66+/hqurq8bxVVZWIiAgANbW1jh9+jR27tyJgwcPNunv0KFDSEtLw6FDh7B161Zs2bKl2RBGRNThCURE1CWMHj1a+Mtf/tLs50+fPi0AEMrLywVBEIRDhw4JAIQ7d+6o2tTU1AimpqbC8ePH1d47d+5c4Y9//KPa+w4ePKj6/M8//ywAEKqrqwVBEIQVK1YIXl5eDx2zvb29sHr1atVrhUIh9OnTR5g8eXKz1+Xi4iK88sorqtc5OTkCAOH9999XHTtx4oQAQMjJyVGNf/78+Wp9//LLL4Kenp5qzPefV6lUCr179xY+//xzQRAE4Y033hCeffZZQalUarwWAMKePXsEQRCEL7/8UrC2thYqKipUn//5558FPT09ITc3VxAEQZg9e7bg4uIi1NXVqdq8+OKLwsyZM5v/ghERdWC8Y0FE1EUlJydj0qRJcHZ2hrm5OUaPHg0AyMrKavY9169fR1VVFcaPHw8zMzPVx7Zt25CWlqbWdtiwYap/t7e3BwBVqZU2SktLkZOTAz8/P9UxAwMD+Pr6PvS99/ZtZ2cHABg6dGiTY43j+e2337Blyxa1awoICIBSqURGRobG80okEshkMtU55syZg5SUFAwYMABvvvkmDhw40Oz4Ll26BC8vL/To0UN17KmnnoJSqVTdAQGAwYMHQ19fX/Xa3t5ep68hEVFHwsXbRERdUGMpTkBAAL7++mvY2toiKysLAQEBqK2tbfZ9jWsAfv75Zzg6Oqp9ztjYWO31vTs1SSQSAA3rM9qDpr4fNJ6Kigq8/vrrePPNN5ucy9nZWeN5G8/TeI7HH38cGRkZ2L9/Pw4ePIgZM2bA399fbf1JS67j/v6IiDobBgsioi7o8uXLKCoqwscffwwnJycAaLKY2cjICEDDAupGnp6eMDY2RlZWluoOx6MwMjJSO68mlpaWsLe3R2JiIkaNGgUAqKurQ3JyMh5//PFH7luTxx9/HKmpqejbt2+LzmNhYYGZM2di5syZeOGFFxAYGIji4uImO1cNGjQIW7ZsQWVlpequxa+//go9PT0MGDCgRWMgIuqoWApFRNQFOTs7w8jICOvXr0d6ejp++OEHrFq1Sq2Ni4sLJBIJfvrpJxQUFKCiogLm5uZYvHgx3n77bWzduhVpaWk4c+YM1q9fj61bt2rdv6urKzIyMpCSkoLCwkLI5XKN7f7yl7/g448/xt69e3H58mX8+c9/RklJSUsuXaO//vWvOH78OEJDQ5GSkoJr167hv//9b5PF1A8SHR2Nb7/9FpcvX8bVq1exc+dOyGQyjc8A+dOf/gSpVIrZs2fjwoULOHToEN544w28+uqrqjItIqKuhsGCiKgLsrW1xZYtW7Bz5054enri448/xt///ne1No6OjoiMjMTSpUthZ2en+iF71apVeP/99xEVFYVBgwYhMDAQP//8M9zc3LTuf/r06QgMDMTYsWNha2uLb7/9VmO7d955B6+++ipmz56NkSNHwtzcHFOnTn30C2/GsGHDcOTIEVy9ehXPPPMMHnvsMSxfvhwODg5an8Pc3ByrV6+Gr68vhg8fjszMTOzbtw96ek3/V2pqaoq4uDgUFxdj+PDheOGFFzBu3Dh89tlnrXlZREQdikQQ7tswnIiIiIiISEe8Y0FERERERC3GYEFERERERC3GYEFERERERC3GYEFERERERC3GYEFERERERC3GYEFERERERC3GYEFERERERC3GYEFERERERC3GYEFERERERC3GYEFERERERC3GYEFERERERC3GYEFERERERC32/7XZ1iHnOxlAAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 800x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Assume 'Ep' (best epoch) has been identified from the previous tuning\n",
    "Ep = 15  # Example value, replace this with the actual best epoch from the epoch tuning step\n",
    "\n",
    "# List of latent dimensions to try\n",
    "latent_dimensions = [1, 3, 5, 7, 9]\n",
    "\n",
    "# Fixed parameters\n",
    "reg = 0.001\n",
    "lr = 0.01\n",
    "\n",
    "# Placeholder for test RMSE values for each latent dimension\n",
    "test_rmse_list = []\n",
    "\n",
    "# Iterate over latent dimensions and train the model\n",
    "for latent in latent_dimensions:\n",
    "    mf_model = MF(train_mat, test_mat, latent=latent, lr=lr, reg=reg)\n",
    "    _, epoch_test_RMSE_list = mf_model.train(epoch=Ep, verbose=False)\n",
    "    test_rmse_list.append(epoch_test_RMSE_list[-1])  # Append RMSE from the best epoch\n",
    "\n",
    "# Plot test RMSE for different latent dimensions\n",
    "fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))\n",
    "ax.plot(latent_dimensions, test_rmse_list, marker='o', linewidth=1.5, markersize=8)\n",
    "ax.set_xticks(latent_dimensions)\n",
    "ax.set_ylabel('test RMSE')\n",
    "ax.set_xlabel('latent dimension')\n",
    "ax.set_title('Tune latent dimension')\n",
    "ax.grid(True)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZN1IW3e19SDY"
   },
   "source": [
    "### Tune regularization weight\n",
    "\n",
    "Last, you can plot how the test RMSE changes when you set different regularization weight. Please run the MF model with 'reg' as {0.0001,0.0005,0.001,0.0015,0.002}, and plot corresponding test RMSE for these five different regularization weights in the next cell.\n",
    "\n",
    "For these five runs of experiments, record the test RMSE after Ep training epochs -- Ep is the best epoch you find by the 'Tune training epoch' plot. And here, fix latent dimension as the one you find the best by the previous part."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 410
    },
    "executionInfo": {
     "elapsed": 1030943,
     "status": "ok",
     "timestamp": 1732819855918,
     "user": {
      "displayName": "NAMITA CHOUGULE",
      "userId": "09547071359313439056"
     },
     "user_tz": 300
    },
    "id": "p6E_wkV39SDZ",
    "outputId": "7057111a-7952-43ee-ba83-65fa7ead77da"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAsUAAAGJCAYAAABiuU6SAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAACCmElEQVR4nO3deXxM1/8G8GdmskcWZBWRiJCIaBASQi0VYmnQqr3ETktblLZ2vooWVa1aSu1qqdLSChGKIhGEEESIWCOLIPs+c35/pObXqSBhkpvleb9eqc6dO/d87txjPLlz7j0yIYQAEREREVEVJpe6ACIiIiIiqTEUExEREVGVx1BMRERERFUeQzERERERVXkMxURERERU5TEUExEREVGVx1BMRERERFUeQzERERERVXkMxURERERU5TEUExFVEDKZDHPmzNHqNjdu3AiZTIbbt29rdbvltd2ycvv2bchkMmzcuPGVX7tkyRLtF0ZEz8VQTERaI5PJivVz7NgxqUtVexpAnv7I5XLUqFEDXbt2RWhoqNTlVXgLFizA77//LnUZam5ubvDw8Hhm+W+//QaZTIZ27do989z69eshk8lw6NChsiixRAIDA7X+ixJRVaUjdQFEVHls2bJF4/HmzZsRHBz8zPKGDRuWZVnFMmDAAHTr1g1KpRLXr1/HypUr0aFDB5w9exaNGzeWurxSM3jwYPTv3x/6+vqlsv0FCxbgvffeQ69evcq03edp06YN1q1bh9TUVJiZmamXnzp1Cjo6Ojh79izy8/Ohq6ur8ZxCoUCrVq2K3Y6DgwOys7M1tlMaAgMDsWLFCgZjIi1gKCYirXn//fc1Hp8+fRrBwcHPLC+PmjVrplHnm2++ia5du2LVqlVYuXKlhJWVjszMTBgbG0OhUEChUJR5+1K126ZNG6xduxYhISHo2rWrevmpU6fQt29fbNu2DeHh4WjZsqX6uZMnT+KNN96AiYlJsduRyWQwMDDQau1EVLo4fIKIypSjoyOGDh36zPL27dujffv26sfHjh2DTCbDL7/8gvnz56N27dowMDBAx44dERMT88zrw8LC0KVLF5iZmcHIyAjt2rXDqVOnXrnON998EwBw8+ZNjeUpKSmYMGEC7O3toa+vD2dnZ3z99ddQqVQa6z169AiDBw+GqakpzM3NERAQgIsXLz4zzvS/+/3U0KFD4ejo+MIa79y5gw8//BAuLi4wNDREzZo10adPn2fG6T4dv3v8+HF8+OGHsLKyQu3atTWee/qaOXPmPHfYy7+P25IlS+Dj44OaNWvC0NAQnp6e+PXXXzXalclkyMzMxKZNm57ZxvPGFK9cuRKNGjWCvr4+atWqhXHjxiElJUVjnfbt28Pd3R1Xr15Fhw4dYGRkBDs7OyxatOiF7xdQGIoBaPSNnJwcnD9/Hu+++y6cnJw0nnv48CGuX7+ufh0AxMXFYfjw4bC2toa+vj4aNWqE9evXa7TzvDHFu3btgpubGwwMDODu7o7ffvvthcd6zZo1qFevHvT19dGiRQucPXtW/dzQoUOxYsUKAJpDl4jo1fBMMRGVa1999RXkcjkmT56M1NRULFq0CIMGDUJYWJh6nb/++gtdu3aFp6cnZs+eDblcjg0bNuCtt97CiRMn4OXlVeJ2n4a16tWrq5dlZWWhXbt2iIuLw5gxY1CnTh2EhIRg6tSpiI+Px7JlywAAKpUK/v7+OHPmDD744AO4urpi7969CAgIeK334r/Onj2LkJAQ9O/fH7Vr18bt27exatUqtG/fHlevXoWRkZHG+h9++CEsLS0xa9YsZGZmFrnNd999F87OzhrLwsPDsWzZMlhZWamXfffdd+jRowcGDRqEvLw87NixA3369MGff/6J7t27AygcTjNy5Eh4eXlh9OjRAIB69eo9d3/mzJmDuXPnwtfXFx988AGio6OxatUqnD17FqdOndIYivDkyRN06dIF7777Lvr27Ytff/0Vn3/+ORo3bqxxBvi/nJycUKtWLZw8eVLjfczLy4OPjw98fHxw6tQpfPrppwCAkJAQAP8fphMTE9GyZUvIZDKMHz8elpaWOHDgAEaMGIG0tDRMmDDhuW3v378f/fr1Q+PGjbFw4UI8efIEI0aMgJ2dXZHrb9u2Denp6RgzZgxkMhkWLVqEd999F7GxsdDV1cWYMWPw4MGDIocoEdErEEREpWTcuHHivx8zDg4OIiAg4Jl127VrJ9q1a6d+fPToUQFANGzYUOTm5qqXf/fddwKAiIyMFEIIoVKpRP369YWfn59QqVTq9bKyskTdunVFp06dXljjrVu3BAAxd+5c8fDhQ5GQkCBOnDghWrRoIQCIXbt2qdedN2+eMDY2FtevX9fYxhdffCEUCoW4e/euEEKI3bt3CwBi2bJl6nWUSqV46623BACxYcOG5+73UwEBAcLBwUFjGQAxe/ZsjX38r9DQUAFAbN68Wb1sw4YNAoBo06aNKCgo0Fj/6XO3bt0q8v15+PChqFOnjmjcuLHIyMh4btt5eXnC3d1dvPXWWxrLjY2Nizze/203KSlJ6Onpic6dOwulUqle74cffhAAxPr169XL2rVr98w+5ubmChsbG9G7d+8i9+Pf+vTpIwwNDUVeXp4QQoiFCxeKunXrCiGEWLlypbCyslKvO3nyZAFAxMXFCSGEGDFihLC1tRXJycka2+zfv78wMzNTvy9P+9W/j3Xjxo1F7dq1RXp6unrZsWPHBACNY/30tTVr1hSPHz9WL9+7d68AIP744w/1sqL+jhHRq+HwCSIq14YNGwY9PT3146fDGmJjYwEAERERuHHjBgYOHIhHjx4hOTkZycnJyMzMRMeOHfH3338/M7ShKLNnz4alpSVsbGzw5ptvIioqCt988w3ee+899Tq7du3Cm2++ierVq6vbSU5Ohq+vL5RKJf7++28AwMGDB6Grq4tRo0apXyuXyzFu3DitvCdPGRoaqv8/Pz8fjx49grOzM8zNzXH+/Pln1h81alSJxvEqlUoMGDAA6enp+O2332BsbFxk20+ePEFqairefPPNItstjsOHDyMvLw8TJkyAXP7//zSNGjUKpqam2L9/v8b61apV0xgDrqenBy8vL3W/eJE2bdogOzsb4eHhAAqHUvj4+AAAWrdujaSkJNy4cUP9XN26dVGrVi0IIbB79274+/tDCKHRB/z8/JCamvrc/X/w4AEiIyMxZMgQVKtWTb28Xbt2z72Qs1+/fhrfVPy37xORdnH4BBGVa3Xq1NF4/DQkPHnyBADU4eVFQxNSU1M1wkVRRo8ejT59+iAnJwd//fUXvv/+eyiVSo11bty4gUuXLsHS0rLIbSQlJQEoHOtra2v7zPCF/w5LeF3Z2dlYuHAhNmzYgLi4OAgh1M+lpqY+s37dunVLtP0ZM2bgr7/+wv79+58Z9vDnn3/iyy+/REREBHJzc9XLX3VM6507dwAALi4uGsv19PTg5OSkfv6p2rVrP9NW9erVcenSpZe29e9xxd7e3ggJCcGXX34JAHB3d4epqSlOnToFe3t7hIeHo1+/fgAKxxenpKRgzZo1WLNmTZHbftoHnrd/RfUBZ2fnIsP0y/o+EWkXQzERlannhSalUlnkWcznndl8GgCfngVevHgxmjRpUuS6/z4z9zz169eHr68vAODtt9+GQqHAF198gQ4dOqB58+bqtjp16oTPPvusyG00aNDgpe38l0wm0wizT/03kBflo48+woYNGzBhwgS0atUKZmZmkMlk6N+/f5Fnx/99dvdlfv/9d3z99deYN28eunTpovHciRMn0KNHD7Rt2xYrV66Era0tdHV1sWHDBmzbtq3YbbyOl/WLF/Hw8ICJiQlOnjyJbt264fHjx+ozxXK5HN7e3jh58iTq1auHvLw8dYh++p6+//77z/0l7I033niV3SnS6+wjEZUcQzERlanq1as/czcBoPBMmpOTU4m39/QMpqmpqTrUasP06dOxdu1azJgxAwcPHlS3lZGR8dJ2HBwccPToUWRlZWmcLS7qrhnVq1cv8uvw/54ZLcqvv/6KgIAAfPPNN+plOTk5Rb6/JXH9+nUEBASgV69emDZt2jPP7969GwYGBggKCtK4z/CGDRueWbe4Z44dHBwAANHR0Rr9IC8vD7du3dLqsVUoFGjZsiVOnTqFkydPwtTUVGMIg4+PD3bu3Kk+q/s0FFtaWsLExARKpbLE9Tzdv6L6QFHLiot3myDSHo4pJqIyVa9ePZw+fRp5eXnqZX/++Sfu3bv3Stvz9PREvXr1sGTJEmRkZDzz/MOHD19pu+bm5hgzZgyCgoIQEREBAOjbty9CQ0MRFBT0zPopKSkoKCgAAPj5+SE/Px9r165VP69SqdS3z/q3evXq4dq1axp1Xrx4sVi3k1MoFM+cNVy+fHmxzjI/T0ZGBt555x3Y2dmpb6VWVLsymUyjndu3bxc5c52xsXGxQrqvry/09PTw/fffa+zT04k2nt7RQlvatGmDhw8fYsOGDfD29tYYx+zj44Po6Gjs3bsXNWvWVE82o1Ao0Lt3b+zevRuXL19+Zpsv6mu1atWCu7s7Nm/erNFPjx8/jsjIyFfej6fjvF/3FyEi4pliIipjI0eOxK+//oouXbqgb9++uHnzJrZu3frCW3W9iFwux08//YSuXbuiUaNGGDZsGOzs7BAXF4ejR4/C1NQUf/zxxytt+5NPPsGyZcvw1VdfYceOHZgyZQr27duHt99+G0OHDoWnpycyMzMRGRmJX3/9Fbdv34aFhQV69eoFLy8vfPrpp4iJiYGrqyv27duHx48fA9A8uzd8+HAsXboUfn5+GDFiBJKSkrB69Wo0atQIaWlpL6zv7bffxpYtW2BmZgY3NzeEhobi8OHDqFmz5ivtLwDMnTsXV69exYwZM7B3716N5+rVq4dWrVqhe/fuWLp0Kbp06YKBAwciKSkJK1asgLOz8zNjej09PXH48GEsXboUtWrVQt26deHt7f1Mu5aWlpg6dSrmzp2LLl26oEePHoiOjsbKlSvRokULrU8A8/Tsb2ho6DOzwT295drp06fh7++vcby++uorHD16FN7e3hg1ahTc3Nzw+PFjnD9/HocPH1Yf46IsWLAAPXv2ROvWrTFs2DA8efIEP/zwA9zd3Yv8ha44PD09AQAff/wx/Pz8oFAo0L9//1faFlGVJ9l9L4io0nve7aK++eYbYWdnJ/T19UXr1q3FuXPnnntLtn/fEk2Iom91JYQQFy5cEO+++66oWbOm0NfXFw4ODqJv377iyJEjL6zx6fYWL15c5PNDhw4VCoVCxMTECCGESE9PF1OnThXOzs5CT09PWFhYCB8fH7FkyRL1Lb6EKLyV2cCBA4WJiYkwMzMTQ4cOFadOnRIAxI4dOzTa2Lp1q3BychJ6enqiSZMmIigoqFi3ZHvy5IkYNmyYsLCwENWqVRN+fn7i2rVrz9z27untz86ePfvM/v331mgBAQECQJE//97munXrRP369YW+vr5wdXUVGzZsELNnz37meF+7dk20bdtWGBoaamzjebeC++GHH4Srq6vQ1dUV1tbW4oMPPhBPnjzRWKddu3aiUaNGz+xLUe/Z82RmZgodHR0BQBw6dOiZ59944w0BQHz99dfPPJeYmCjGjRsn7O3tha6urrCxsREdO3YUa9asUa/zvH66Y8cO4erqKvT19YW7u7vYt2+f6N27t3B1dX3mtUX1yf/2gYKCAvHRRx8JS0tLIZPJeHs2otcgE4Ij9omIysLvv/+Od955BydPnkTr1q2lLofKiSZNmsDS0hLBwcFSl0JUpXFMMRFRKcjOztZ4rFQqsXz5cpiamqJZs2YSVUVSys/PV487f+rYsWO4ePFikVN9E1HZ4phiIqJS8NFHHyE7OxutWrVCbm4u9uzZg5CQECxYsKBEt0ajyiMuLg6+vr54//33UatWLVy7dg2rV6+GjY0Nxo4dK3V5RFUeh08QEZWCbdu24ZtvvkFMTAxycnLg7OyMDz74AOPHj5e6NJJIamoqRo8ejVOnTuHhw4cwNjZGx44d8dVXX73yhaZEpD0MxURERERU5XFMMRERERFVeQzFRERERFTl8UK7V6RSqfDgwQOYmJhwmk0iIiKickgIgfT0dNSqVUtj5sqiMBS/ogcPHsDe3l7qMoiIiIjoJe7du4fatWu/cB2G4ldkYmICoPBNNjU1LfX28vPzcejQIXTu3Bm6urql3h6VHR5b0ib2J9I29inSprLuT2lpabC3t1fnthdhKH5FT4dMmJqallkoNjIygqmpKT+UKhkeW9Im9ifSNvYp0iap+lNxhrryQjsiIiIiqvIYiomIiIioymMoJiIiIqIqj6GYiIiIiKo8hmIiIiIiqvJ494lyLidficDIeBy8HI/Y+3LsT41AF3dbdGtsCwNdhdTlEREREVUKDMXlWPDVRHy6KwJp2QWQywCVkCM2KgmHriZhzh9XsLRPE/i6WUtdJhEREVGFx+ET5VTw1USM3nIO6dkFAACVgMaf6dkFGLXlHIKvJkpUIREREVHlwVBcDuXkK/HprghAAOI564h//jN5VwRy8pVlVxwRERFRJcRQXA4FRsYjLbvguYH4KQEgNbsABy7Hl0VZRERERJUWQ3E5dOhKIuQvn40QACCXAUGXOYSCiIiI6HUwFJdDKVl56rHDL6MSQEp2XukWRERERFTJMRSXQ+ZGeiU6U2xuqFe6BRERERFVcgzF5VDnRtYlOlPs587bshERERG9DobicqhbY1uYGurgZSeLZQDMDHXQ1d22LMoiIiIiqrQYisshA10FlvZpAsjw0mD8TZ8mnNmOiIiI6DUxFJdTvm7WWDO4OUwNCycdfDrG+N9jjd+sb8EZ7YiIiIi0gKG4HOvkZo2wab74tp8HfBtawdlUBd+GVvj4LWcAwMmYZEQnpEtcJREREVHFx1BczhnoKvBO09pYMaAJPmqkwooBTTCpswu6uttAJYCFB6KkLpGIiIiowmMorqA+6+IKHbkMx6If4sSNh1KXQ0RERFShMRRXUHUtjPF+SwcAwPz9UVAW9x5uRERERPQMhuIK7OOO9WFioINrCenYc/6+1OUQERERVVgMxRVYDWM9jO9QeNHdkkPRyM5TSlwRERERUcXEUFzBBfg4ws7cEIlpufjpRKzU5RARERFVSAzFFZyBrgKfdXEBAKw+fhMP03MlroiIiIio4mEorgT836gFj9pmyMxTYtnh61KXQ0RERFThMBRXAnK5DNO6NQQA7Dh7DzcSOaEHERERUUkwFFcS3k410dnNGkqVwFcHrkldDhEREVGFwlBciXze1RUKuQxHriUhJCZZ6nKIiIiIKgyG4kqknmU1DPKuAwCYHxgFFSf0ICIiIioWhuJK5pOO9VFNXwdXHqTh94g4qcshIiIiqhAYiiuZmtX08WGHegCAxUHRyMnnhB5EREREL8NQXAkNb10XtcwMEJ+ag3Unb0ldDhEREVG5x1BcCRnoKjDlnwk9Vh27ieQMTuhBRERE9CIMxZVUTw87uNuZIiO3AN8fuSF1OURERETlGkNxJfXvCT1+DruLmw8zJK6IiIiIqPxiKK7EfOpZwLehFSf0ICIiInoJhuJK7ot/JvQIvpqI07GPpC6HiIiIqFxiKK7knK1M0L+FPQBgASf0ICIiIioSQ3EVMMG3AYz1FLh0PxV/XHogdTlERERE5Q5DcRVgaaKPD9oXTuix6CAn9CAiIiL6L4biKmJEGyfYmBogLiUbG0NuS10OERERUbnCUFxFGOopMNmvcEKPFX/F4HFmnsQVEREREZUfDMVVyDtN7eBma4p0TuhBREREpEHyULxixQo4OjrCwMAA3t7eOHPmzAvXX7ZsGVxcXGBoaAh7e3tMnDgROTk56ufT09MxYcIEODg4wNDQED4+Pjh79qzGNhITEzF06FDUqlULRkZG6NKlC27cqPwhUSGXYXr3wgk9tp6+g1vJmRJXRERERFQ+SBqKd+7ciUmTJmH27Nk4f/48PDw84Ofnh6SkpCLX37ZtG7744gvMnj0bUVFRWLduHXbu3Ilp06ap1xk5ciSCg4OxZcsWREZGonPnzvD19UVcXBwAQAiBXr16ITY2Fnv37sWFCxfg4OAAX19fZGZW/pDY2tkC7V0sUaAS+JoTehAREREBkDgUL126FKNGjcKwYcPg5uaG1atXw8jICOvXry9y/ZCQELRu3RoDBw6Eo6MjOnfujAEDBqjPLmdnZ2P37t1YtGgR2rZtC2dnZ8yZMwfOzs5YtWoVAODGjRs4ffo0Vq1ahRYtWsDFxQWrVq1CdnY2tm/fXmb7LqWpXRtCLgMOXknA2duPpS6HiIiISHI6UjWcl5eH8PBwTJ06Vb1MLpfD19cXoaGhRb7Gx8cHW7duxZkzZ+Dl5YXY2FgEBgZi8ODBAICCggIolUoYGBhovM7Q0BAnT54EAOTm5gKAxjpyuRz6+vo4efIkRo4cWWTbubm56tcCQFpaGgAgPz8f+fn5Jd39EnvahjbacqppgD6edth5Lg7z/ryCX0d7QyaTvfZ26dVo89gSsT+RtrFPkTaVdX8qSTuSheLk5GQolUpYW1trLLe2tsa1a0V/rT9w4EAkJyejTZs2EEKgoKAAY8eOVQ+fMDExQatWrTBv3jw0bNgQ1tbW2L59O0JDQ+Hs7AwAcHV1RZ06dTB16lT8+OOPMDY2xrfffov79+8jPj7+ufUuXLgQc+fOfWb5oUOHYGRk9KpvQ4kFBwdrZTuNBKAnV+DS/TTM33IQzSw4053UtHVsiQD2J9I+9inSprLqT1lZWcVeV7JQ/CqOHTuGBQsWYOXKlfD29kZMTAw++eQTzJs3DzNnzgQAbNmyBcOHD4ednR0UCgWaNWuGAQMGIDw8HACgq6uLPXv2YMSIEahRowYUCgV8fX3RtWtXCPH8YDh16lRMmjRJ/TgtLQ329vbo3LkzTE1NS3fHUfibTnBwMDp16gRdXV2tbDPZ7Ca+/+smjjw0xpSBbaCvI/l1l1VSaRxbqrrYn0jb2KdIm8q6Pz39Zr84JAvFFhYWUCgUSExM1FiemJgIGxubIl8zc+ZMDB48WD3EoXHjxsjMzMTo0aMxffp0yOVy1KtXD8ePH0dmZibS0tJga2uLfv36wcnJSb0dT09PREREIDU1FXl5ebC0tIS3tzeaN2/+3Hr19fWhr6//zHJdXd0y/ZDQZntj2ztjx9n7uJ+Sg+1n4zCqrdPLX0Slpqz7ElVu7E+kbexTpE1l1Z9K0oZkpwb19PTg6emJI0eOqJepVCocOXIErVq1KvI1WVlZkMs1S1YoFADwzFleY2Nj2Nra4smTJwgKCkLPnj2f2Z6ZmRksLS1x48YNnDt3rsh1KjMjPR1M7lw4ocfyv24gJYsTehAREVHVJOn35ZMmTcLatWuxadMmREVF4YMPPkBmZiaGDRsGABgyZIjGhXj+/v5YtWoVduzYgVu3biE4OBgzZ86Ev7+/OhwHBQXh4MGD6uc7dOgAV1dX9TYBYNeuXTh27Jj6tmydOnVCr1690Llz57J9A8qB3p614WpjgrScAiz/K0bqcoiIiIgkIemY4n79+uHhw4eYNWsWEhIS0KRJExw8eFB98d3du3c1zgzPmDEDMpkMM2bMQFxcHCwtLeHv74/58+er10lNTcXUqVNx//591KhRA71798b8+fM1Tp/Hx8dj0qRJSExMhK2tLYYMGaIek1zVKOQyTOvWEEPWn8Hm0NsY0soBDjWNpS6LiIiIqEzJxIuuLqPnSktLg5mZGVJTU8vsQrvAwEB069atVMbgDF4XhhM3ktG9sS1WDGqm9e3T85X2saWqhf2JtI19irSprPtTSfIabzdAAIBp3RpCJgP2R8Yj/M4TqcshIiIiKlMMxQQAaGhrij6etQEA8/dffeHt6YiIiIgqG4ZiUpvUyQWGugqcv5uCA5cTpC6HiIiIqMwwFJOajZmB+l7FXx24hrwClcQVEREREZUNhmLSMKatEyyq6ePu4yxsOX1H6nKIiIiIygRDMWkw1tfBp50bAAC+P3IDqVn5EldEREREVPoYiukZfTxro4F1NaRm52PFMU7oQURERJUfQzE9Q0chx9RuDQEAG0/dxr3HWRJXRERERFS6GIqpSO0bWKK1c03kKVVYFBQtdTlEREREpYqhmIokk8nUE3r8cfEBLtzlhB5ERERUeTEU03M1qmWGd5sWTuixIDCKE3oQERFRpcVQTC802a8B9HXkOHv7CYKuJEpdDhEREVGpYCimF7I1M8SoN59O6BHFCT2IiIioUmIoppca274eLKrp4fajLGwL44QeREREVPkwFNNLVdPXwQTfwgk9vjtyA6nZnNCDiIiIKheGYiqW/i3sUc/SGE+y8rHq2E2pyyEiIiLSKoZiKhYdhRzT/pnQY/2pW7j/hBN6EBERUeXBUEzF9parFVo61UBegQpLOKEHERERVSIMxVRsMpkM07u5AQB+j3iAS/dTpC2IiIiISEsYiqlEGtc2wztN7QAA8/dzQg8iIiKqHBiKqcQm+7lAT0eOsFuPcTgqSepyiIiIiF4bQzGVmJ25IUa0qQsAWHggCvlKTuhBREREFRtDMb2SD9rXQw1jPcQ+zMSOM3elLoeIiIjotTAU0ysxNdDFBN/6AIBlh28gPYcTehAREVHFxVBMr2yAVx04WRjjUWYeVh/nhB5ERERUcTEU0yvTVcjxeVdXAMBPJ27hQUq2xBURERERvRqGYnotnd2s4eVYA7kFKiw5xAk9iIiIqGJiKKbXIpPJMK174fTPv12Iw+W4VIkrIiIiIio5hmJ6bU3szdHDoxaE4IQeREREVDExFJNWTPFzgZ5CjtDYRzgazQk9iIiIqGJhKCatsK9hhGGtHQEACwKvoYATehAREVEFwlBMWvNhB2eYG+kiJikDO8/dk7ocIiIiomJjKCatMTPUxScdCyf0+Db4BjJyCySuiIiIiKh4GIpJqwZ5O8CxphGSM3KxhhN6EBERUQXBUExapacjx+ddCif0WHMiFgmpORJXRERERPRyDMWkdV3cbeDpUB05+Sp8wwk9iIiIqAJgKCatk8lkmP7PhB6/nr+Pqw/SJK6IiIiI6MUYiqlUNKtTHd3fsIUQwIJATuhBRERE5RtDMZWaz/1coauQ4WRMMo5ffyh1OURERETPxVBMpaZOTSMEtHIEUHi2mBN6EBERUXnFUEylavxbzjAz1MX1xAz8Gn5f6nKIiIiIisRQTKXK3EgPH73lDABYGnwdmZzQg4iIiMohhmIqdYNbOaBODSMkpedi7YlYqcshIiIiegZDMZU6fR0FPuviAgD48XgsktI4oQcRERGVLwzFVCa6N7ZFE3tzZOcrsTT4utTlEBEREWmQPBSvWLECjo6OMDAwgLe3N86cOfPC9ZctWwYXFxcYGhrC3t4eEydORE7O/595TE9Px4QJE+Dg4ABDQ0P4+Pjg7NmzGtvIyMjA+PHjUbt2bRgaGsLNzQ2rV68ulf2jQjKZDDP+mdDjl3P3cC2BE3oQERFR+SFpKN65cycmTZqE2bNn4/z58/Dw8ICfnx+SkpKKXH/btm344osvMHv2bERFRWHdunXYuXMnpk2bpl5n5MiRCA4OxpYtWxAZGYnOnTvD19cXcXFx6nUmTZqEgwcPYuvWrYiKisKECRMwfvx47Nu3r9T3uSpr7lgDXd1toBLAwsBrUpdDREREpCZpKF66dClGjRqFYcOGqc/WGhkZYf369UWuHxISgtatW2PgwIFwdHRE586dMWDAAPXZ5ezsbOzevRuLFi1C27Zt4ezsjDlz5sDZ2RmrVq3S2E5AQADat28PR0dHjB49Gh4eHi89S02v7/MurtCRy3D8+kP8zQk9iIiIqJzQkarhvLw8hIeHY+rUqeplcrkcvr6+CA0NLfI1Pj4+2Lp1K86cOQMvLy/ExsYiMDAQgwcPBgAUFBRAqVTCwMBA43WGhoY4efKkxnb27duH4cOHo1atWjh27BiuX7+Ob7/99rn15ubmIjc3V/04La3w6//8/Hzk5+eX/A0ooadtlEVbpcnOTA+DvO2xKfQuFuy/Ci+HVlDIZVKXJanKcmypfGB/Im1jnyJtKuv+VJJ2JAvFycnJUCqVsLa21lhubW2Na9eK/mp94MCBSE5ORps2bSCEQEFBAcaOHasePmFiYoJWrVph3rx5aNiwIaytrbF9+3aEhobC2dlZvZ3ly5dj9OjRqF27NnR0dCCXy7F27Vq0bdv2ufUuXLgQc+fOfWb5oUOHYGRk9CpvwSsJDg4us7ZKi0s+YKhQ4FpiBuZuPoiWVkLqksqFynBsqfxgfyJtY58ibSqr/pSVlVXsdSULxa/i2LFjWLBgAVauXAlvb2/ExMTgk08+wbx58zBz5kwAwJYtWzB8+HDY2dlBoVCgWbNmGDBgAMLDw9XbWb58OU6fPo19+/bBwcEBf//9N8aNG4datWrB19e3yLanTp2KSZMmqR+npaXB3t4enTt3hqmpaenuOAp/0wkODkanTp2gq6tb6u2VttSat/F10HX8lWSEqQPbwFBPIXVJkqlsx5akxf5E2sY+RdpU1v3p6Tf7xSFZKLawsIBCoUBiYqLG8sTERNjY2BT5mpkzZ2Lw4MEYOXIkAKBx48bIzMzE6NGjMX36dMjlctSrVw/Hjx9HZmYm0tLSYGtri379+sHJyQlA4bjjadOm4bfffkP37t0BAG+88QYiIiKwZMmS54ZifX196OvrP7NcV1e3TD8kyrq90jKsjRN+PnMP959kY9Ppe/ioY32pS5JcZTm2VD6wP5G2sU+RNpVVfypJG5JdaKenpwdPT08cOXJEvUylUuHIkSNo1apVka/JysqCXK5ZskJReIZRCM2v4I2NjWFra4snT54gKCgIPXv2BPD/Y4CL2o5KpXrt/aLiMdBVYIpf4YQeq47fRFI6J/QgIiIi6Ug6fGLSpEkICAhA8+bN4eXlhWXLliEzMxPDhg0DAAwZMgR2dnZYuHAhAMDf3x9Lly5F06ZN1cMnZs6cCX9/f3U4DgoKghACLi4uiImJwZQpU+Dq6qrepqmpKdq1a4cpU6bA0NAQDg4OOH78ODZv3oylS5dK80ZUUf5v1ML6k7dw8X4qlh2+gQXvNJa6JCIiIqqiJA3F/fr1w8OHDzFr1iwkJCSgSZMmOHjwoPriu7t372qc0Z0xY0bhJBAzZiAuLg6Wlpbw9/fH/Pnz1eukpqZi6tSpuH//PmrUqIHevXtj/vz5GqfPd+zYgalTp2LQoEF4/PgxHBwcMH/+fIwdO7bsdp4gl8swrVtD9FtzGjvO3MUwH0fUtzaRuiwiIiKqgiS/0G78+PEYP358kc8dO3ZM47GOjg5mz56N2bNnP3d7ffv2Rd++fV/Ypo2NDTZs2FDiWkn7vJ1qorObNQ5dTcTCA9ewfmgLqUsiIiKiKkjyaZ6JvuhaOKHHX9eScComWepyiIiIqApiKCbJOVlWwyDvOgCA+fujoFLxvsVERERUthiKqVz4uGN9mOjr4Gp8Gn67ECd1OURERFTFMBRTuVCzmj4+7FA46+CSQ9HIyVdKXBERERFVJQzFVG4Ma+0IO3NDxKfmYN3JW1KXQ0RERFUIQzGVGwa6Ckz2awAAWHXsJpIzciWuiIiIiKoKhmIqV3p62MHdzhQZuQX47vANqcshIiKiKqLYobhbt25ITU1VP/7qq6+QkpKifvzo0SO4ublptTiqep5O6AEA287cRUxShsQVERERUVVQ7FAcFBSE3Nz//zp7wYIFePz4sfpxQUEBoqOjtVsdVUk+9Szg29AKSpXAVweuSV0OERERVQHFDsVCiBc+JtKmL7q6QiGX4XBUIkJvPpK6HCIiIqrkOKaYyiVnKxMM8LIHACwI5IQeREREVLqKHYplMhlkMtkzy4hKywTfBqimr4PIuFTsu/hA6nKIiIioEtMp7opCCAwdOhT6+voAgJycHIwdOxbGxsYAoDHemEgbLKrp44P29bA4KBqLg6LRxd0GBroKqcsiIiKiSqjYoTggIEDj8fvvv//MOkOGDHn9ioj+ZXjrutgSegdxKdnYGHIbY9vVk7okIiIiqoSKHYo3bNhQmnUQFclQT4HJfi6YvOsiVvwVg77N7VHDWE/qsoiIiKiSee0L7e7cuYOrV69CpVJpox6iZ7zT1A5utqZIzy3A90c4oQcRERFpX7FD8fr167F06VKNZaNHj4aTkxMaN24Md3d33Lt3T+sFEinkMkzvXjihx9bTdxD7kBN6EBERkXYVOxSvWbMG1atXVz8+ePAgNmzYgM2bN+Ps2bMwNzfH3LlzS6VIotbOFujgYokClcDXBzmhBxEREWlXsUPxjRs30Lx5c/XjvXv3omfPnhg0aBCaNWuGBQsW4MiRI6VSJBEATO3WEHIZEHQlEWduPX75C4iIiIiKqdihODs7G6ampurHISEhaNu2rfqxk5MTEhIStFsd0b80sDZBvxZ1AADz91/lhB5ERESkNcUOxQ4ODggPDwcAJCcn48qVK2jdurX6+YSEBJiZmWm/QqJ/mdipPoz0FLh4PxV/RsZLXQ4RERFVEsUOxQEBARg3bhzmzZuHPn36wNXVFZ6enurnQ0JC4O7uXipFEj1lZWKgvlfxooPXkFuglLgiIiIiqgyKfZ/izz77DFlZWdizZw9sbGywa9cujedPnTqFAQMGaL1Aov8a+WZdbD19B/efZGNzyB2MauskdUlEksvJVyIwMh4HL8cj9r4c+1Mj0MXdFt0a23ImSCKiYih2KJbL5fjf//6H//3vf0U+/9+QTFRajPR0MLmzCz7bfQnL/7qB9zxrozon9KAqLPhqIj7dFYG07ALIZYBKyBEblYRDV5Mw548rWNqnCXzdrKUuk4ioXHvtyTuIpNDbszZcbUyQllOA5X/FSF0OkWSCryZi9JZzSM8uAAA8vf706Z/p2QUYteUcgq8mSlQhEVHFUOwzxU5OxfuKOjY29pWLISouhVyGad0aYsj6M9hy+jaGtHKAo4Wx1GURlamcfCU+3RUBCOB592IRAGQCmLwrAmHTfDmUgojoOYodim/fvg0HBwcMHDgQVlZWpVkTUbG0bWCJtg0s8ff1h1gUdA0rB3m+/EVElUhgZDzS/jlD/CICQGp2AQ5cjsc7TWuXfmFERBVQsUPxzp071VM9d+3aFcOHD0e3bt0gl3MEBklnWjdXnLzxEIGRCQi/8xieDjWkLomozBy6kvjPGOKXryuXAUGXExmKiYieo9iJtk+fPjhw4ABiYmLg6emJiRMnwt7eHl988QVu3LhRmjUSPZerjSn6eNoDAL7cHwUhOKEHVR0pWXnFCsRAYXBOyc4r3YKIiCqwEp/mtbOzw/Tp03Hjxg1s27YNYWFhcHV1xZMnT0qjPqKXmtS5AQx1FbhwNwWBkZxVkaoOcyM9yGTFW1cuA8wNeZcWIqLneaWxDzk5Odi6dSvmzp2LsLAw9OnTB0ZGRtqujahYrE0NMPqfexV/ffAa8gpUEldEVPpSs/ORlV+A4n45ohKAnztvy0ZE9DwlCsVhYWEYPXo0bGxssHTpUrz77ruIi4vDjh07oK+vX1o1Er3U6LZOsDTRx93HWdhy+o7U5RCVGqVK4OewO+iw5Bj+vp5c7NeZGuigq7ttKVZGRFSxFTsUN2rUCG+//TYMDQ1x/PhxnD9/HuPHj0f16tVLsz6iYjHW18GkTg0AAN8fuYHUrHyJKyLSvtCbj/D28pOY/ttlPM7MQz1LY3zSsT5kMuBloyjqWVWDQl7MsRZERFVQsUNxVFQUcnJysHnzZnTo0AE1atQo8odIKn08a6OBdTWkZufjh6O8+JMqj3uPszB2SzgGrD2NqPg0mBroYLa/Gw5OaIuJnRpgzeDmMDUsvJnQ09z79E8jPQV05DJcuJuCCTsiUKDk8CIioqIU+5ZsGzZsKM06iF6bjkKOqd0aYtiGs9gUcgeDWzqiTk2OdaeKKzO3ACuPxWDtiVvIK1BBLgMGeTtgYqcGqPGvqc07uVkjbJovDlyOx4HIeMTeT4BTbRt0bWyLru62CL35CKO3nMP+yHjoKGRY2rcJzxoTEf1HsUNxQEBAadZBpBXtG1iijbMFTsYkY1HQNfwwsJnUJRGVmEolsOdCHBYdvIak9FwAgE+9mpjl7wZXG9MiX2Ogq8A7TWvjbXdrBAYGolu3JtDV1QUAdHC1wspBnvhgazj2RjyAQi7D4vc8GIyJiP5FazNvxMfHY/z48draHNErkclkmNrNFTIZ8OeleJy/y1sFUsUSfucJ3ll5CpN3XURSei7q1DDCj4M98fNI7+cG4uLo5GaN5QOaQiGXYc/5OEzdcwmq4t7kmIioCihRKL5y5Qp++OEHrFmzBikpKQCA5ORkTJw4EU5OTjh69Ghp1EhUIo1qmaF3s8JZuxZwQg+qIOJTszFhxwX0XhWCi/dTYaynwBddXRE8qS38GtlAVtwbEr9A18a2WNavCeQy4Jdz9zFj72X+/SAi+kexh0/s27cP7733HgoKCgAAixYtwtq1a9G3b194enrit99+Q5cuXUqtUKKS+LRzA/x56QHO3XmCoCsJ6MJbUVE5lZOvxJq/Y7Hq2E1k5yshkxVeNDrZzwVWJgZab8/foxaUKoGJv0RgW9hd6MplmNOjkVZCNxFRRVbsM8Vffvklxo0bh7S0NCxduhSxsbH4+OOPERgYiIMHDzIQU7lia2aIUW8WTujx1QFO6EHljxACf156gI7fHMfS4OvIzleiuUN17BvXBove8yiVQPxUr6Z2WPyeB2QyYFPoHU6RTkSEEoTi6OhojBs3DtWqVcNHH30EuVyOb7/9Fi1atCjN+ohe2Zh29WBRTQ+3H2VhWxgn9KDy43JcKvr9eBrjt11AXEo2apkZ4PsBTbFrbCs0rm1WJjW851kbX73bGACw7uQtfHXwGoMxEVVpxQ7F6enpMDUtvMhDoVDA0NAQTk5OpVYY0euqpq+DCb6FE3p8d+QGUrM5oQdJ62F6Lj7/9RL8fziJM7cfw0BXjgm+9XHk0/bo4VGrzIcw9GtRB1/2cgcA/Hg8Ft8cus5gTERVVrHHFANAUFAQzMwKz2KoVCocOXIEly9f1linR48e2quO6DX1b2GPDadu4ebDTKw8FoOpXRtKXRJVQbkFSmw8dRvL/4pBRm7hdRk9PGrhi66uqGVuKGlt77d0QIFShTl/XMUPR2Ogo5Cpf5kkIqpKShSK/3uv4jFjxmg8lslkUCqVr18VkZboKOSY1q0hRmw6hw2nbuN9bwfY1+CEHlQ2hBA4HJWEL/dfxZ1HWQCAxnZmmO3vhuaO5WcG0KGt66JAJfDl/igsO3wDugo5xnVwlrosIqIyVexQrFLxQiWqmN5ytUIrp5oIjX2EJYei8V3/plKXRFXA9cR0zPvzKk7cSAYAWJro4zM/F/RuVhvycjhpxsg3nZCvFPj64DUsDoqGjlyGMe3qSV0WEVGZ0drkHa9jxYoVcHR0hIGBAby9vXHmzJkXrr9s2TK4uLjA0NAQ9vb2mDhxInJyctTPp6enY8KECXBwcIChoSF8fHxw9uxZjW3IZLIifxYvXlwq+0jSkclkmN69cNjE3ogHuHgvRdqCqFJ7kpmHWXsvo+t3J3DiRjL0FHJ80L4ejk5ujz7N7ctlIH7qg/b18GmnwqETCw9cw08nYiWuiIio7Egeinfu3IlJkyZh9uzZOH/+PDw8PODn54ekpKQi19+2bRu++OILzJ49G1FRUVi3bh127tyJadOmqdcZOXIkgoODsWXLFkRGRqJz587w9fVFXFycep34+HiNn/Xr10Mmk6F3796lvs9U9tztzPBuUzsAwPxA3n6KtC9fqcLGU7fQfskxbA69A6VKwK+RNQ5PaofPu7iimn6JRqtJ5qOO9fFxx/oAgC/3R2Fz6G1pCyIiKiOSh+KlS5di1KhRGDZsGNzc3LB69WoYGRlh/fr1Ra4fEhKC1q1bY+DAgXB0dETnzp0xYMAA9dnl7Oxs7N69G4sWLULbtm3h7OyMOXPmwNnZGatWrVJvx8bGRuNn79696NChA++oUYl96ucCfR05ztx6jMNRRf/SRfQq/r7+EN2+O4E5f1xFanY+XG1MsG2kN34c3Bx1ala8MewTfevjw/aFQydm7b2CbWF3Ja6IiKj0SXrqIi8vD+Hh4Zg6dap6mVwuh6+vL0JDQ4t8jY+PD7Zu3YozZ87Ay8sLsbGxCAwMxODBgwEABQUFUCqVMDDQvPG9oaEhTp48WeQ2ExMTsX//fmzatOm5tebm5iI3N1f9OC0tDQCQn5+P/PzSv9XX0zbKoq3KyspYB0NbOeDHE7ewMPAqWjuZQ1ch+e+FPLYV2O1HmVh44Dr+in4IAKhupIsJHZ3R19MOOgq5JMdUW/1pwltOyM0vwLpTdzDtt0hAqNDH004bJVIFw88o0qay7k8laUfSUJycnAylUglra2uN5dbW1rh27VqRrxk4cCCSk5PRpk0bCCFQUFCAsWPHqodPmJiYoFWrVpg3bx4aNmwIa2trbN++HaGhoXB2Lvpq6k2bNsHExATvvvvuc2tduHAh5s6d+8zyQ4cOwcio7M4EBQcHl1lblVHdAsBYR4HY5CzM2hSEN23KzzAKHtuKI7sAOHRfjuMJMiiFDHKZwJs2Al1qZ8MoORKHgiKlLlEr/amxANrZyHE8QY7pv1/GlcuX4GVZfv7OUNniZxRpU1n1p6ysrGKvW+JQ7OTkhLNnz6JmzZoay1NSUtCsWTPExpbuhRnHjh3DggULsHLlSnh7eyMmJgaffPIJ5s2bh5kzZwIAtmzZguHDh8POzg4KhQLNmjXDgAEDEB4eXuQ2169fj0GDBj1zdvnfpk6dikmTJqkfp6Wlwd7eHp07d1ZPalKa8vPzERwcjE6dOkFXV7fU26vMsq3vYu6f13Ak0QDTBraBiYG07yePbcWhVAnsPh+HpYdj8CgzDwDQrr4FpnZ1QT1LY4mrK6Tt/tRNCMz98xp+PnMP228q0KxJY/TwsNVCpVRR8DOKtKms+9PTb/aLo8Sh+Pbt20Xeizg3N1fjQrbisLCwgEKhQGJiosbyxMRE2NjYFPmamTNnYvDgwRg5ciQAoHHjxsjMzMTo0aMxffp0yOVy1KtXD8ePH0dmZibS0tJga2uLfv36FTle+MSJE4iOjsbOnTtfWKu+vj709fWfWa6rq1umHxJl3V5l9H6ruthy+h5ikzPx06m7+KyLq9QlAeCxLe/CYh9h7h9XcTW+8APWydIYM7u7oYOrlcSVFU2b/Wler8ZQAdh+5h6m7I6EgZ4uur/BYFzV8DOKtKms+lNJ2ih2KN63b5/6//89sx0AKJVKHDlyBI6OjsVuGAD09PTg6emJI0eOoFevXgD+f6a88ePHF/marKwsyOWa40AVCgUAPHNHAWNjYxgbG+PJkycICgrCokWLntneunXr4OnpCQ8PjxLVThWXrkKOL7q6YvSWcKw7eQuDWjrATuJZxaj8uvc4C18duIb9kfEAABMDHXzSsT6GtHKEno70Y9LLglwuw/xejVGgFNgVfh8f77gAhVyGLu5Fn7wgIqqIih2Kn4ZWmUz2zMx2urq6cHR0xDfffFPiAiZNmoSAgAA0b94cXl5eWLZsGTIzMzFs2DAAwJAhQ2BnZ4eFCxcCAPz9/bF06VI0bdpUPXxi5syZ8Pf3V4fjoKAgCCHg4uKCmJgYTJkyBa6uruptPpWWloZdu3a9Ut1UsXVys4ZX3Ro4c+sxvgmKxtJ+TaQuicqZrLwCrDp2Ez/+HYu8AhXkMmCAVx1M6tQANas9+61RZSeXy/BV7zegVAnsuRCHj7afx6pBnvB1s375i4mIKoASz2hXt25dnD17FhYWFlopoF+/fnj48CFmzZqFhIQENGnSBAcPHlRffHf37l2NM8MzZsyATCbDjBkzEBcXB0tLS/j7+2P+/PnqdVJTUzF16lTcv38fNWrUQO/evTF//vxnTqHv2LEDQggMGDBAK/tCFYdMJsP0bg3Rc8Up7LkQh+Ft6sLdzuzlL6RKT6US2HsxDl8fiEZCWuGkQK2camKWvxsa2pb+9QPlmUIuw+I+HshXCfxx8QE+/Pk8fhziiQ4u5XMICRFRSciEFmYxSElJgbm5uRbKqTjS0tJgZmaG1NTUMrvQLjAwEN26deOYLi36ZMcF7I14gFZONbFtlDdksrKfbYzHtvyIuJeCuX9cwYW7KQAA+xqGmN6tIfwa2UjSN15FWfSnAqUKH++4gMDIBOjpyLEuoDnerG9ZKm2R9PgZRdpU1v2pJHmtxAPivv76a42L0vr06YMaNWrAzs4OFy9eLHm1RBKa3NkFejpyhMY+wtFoTuhRVSWm5WDSzgj0WnEKF+6mwEhPgSl+Lgie2A5d3G0rTCAuKzoKOb7r3xSd3ayRV6DCyE3nEHIzWeqyiKgcy8lXYs/5+xi3PQLLr8gxbnsE9py/j5z8Z2/eIJUSh+LVq1fD3t4eQOE95g4fPoyDBw+ia9eumDJlitYLJCpN9jWMMMzHEQCwIPAaCpQqaQuiMpWTr8QPf91AhyXHsOdC4d1zejerjaOT22NcB2cY6CokrrD80lXI8cPAZujoaoXcAhVGbDyHsNhHUpdFROVQ8NVEeC04jEm/XMThqCTEpMlxOCoJk365CK8Fh3H4auLLN1IGShyKExIS1KH4zz//RN++fdG5c2d89tlnOHv2rNYLJCptH3ZwhrmRLmKSMrDz3D2py6EyIIRAYGQ8fJcex5JD15GVp0SzOubYO641vunrAWvT59+znP6fno4cK99vhnYNLJGdr8SwjWcRfuex1GURUTkSfDURo7ecQ3p2AQBA9c+g3ad/pmcXYNSWcwguB8G4xKG4evXquHevMDgcPHgQvr6+AAr/kSnq/sVE5Z2ZoS4+6VgfAPBt8HVk5BZIXBGVpisPUtF/zWl8+PN53H+SDVszA3zXvwl2f+ADD3tzqcurcPR1FPhxsCfaOFsgK0+JgPVnceHuE6nLIqJyICdfiU93RQACeN4FbOKf/0zeFSH5UIoSh+J3330XAwcORKdOnfDo0SN07doVAHDhwoXnTqNMVN4N8naAY00jJGfk4cfjN6Uuh0pBckYupu6JxNvLTyLs1mPo68jxccf6OPJpO/RsYsdxw6/BQFeBtUOao6VTDWTkFmDI+jO4dD9F6rKISGKBkfFIyy54biB+SgBIzS7AgcvxZVHWc5U4FH/77bcYP3483NzcEBwcjGrVqgEA4uPj8eGHH2q9QKKyoKdTOKEHAKw9EYv41GyJKyJtyStQ4acTseiw+Bi2n7kLIYC337DFkU/bYVKnBjDSK/HEnlQEQz0F1gW0QAvH6kjPKcDgdWdwOS5V6rKISEKHriRCXszzDXIZEHRZ2iEUJf7XQFdXF5MnT35m+cSJE7VSEJFU/BrZoLlDdZy78wTfHLqOJX04y2FFJoTAX9eSMH9/FGKTMwEA7nammPV2I3jVrSFxdZWTsb4ONgzzwpB1YTh/NwWD14Vh++iWcLWp2vd3JqqqUrLy1GOHX0YlgJTsvNIt6CVeaY7SLVu2oE2bNqhVqxbu3LkDAFi2bBn27t2r1eKIypJMJsP07g0BALvP38eVBzzLVVHFJKUjYMNZjNh0DrHJmbCopodFvd/A3nFtGIhLWTV9HWwc7gUPe3M8ycrHoLVhuJGYLnVZRCQBcyM9FHdkmlwGmBvqlW5BL6uhpC9YtWoVJk2ahK5duyIlJUV9cZ25uTmWLVum7fqIylTTOtXx9hu2EAJYGHgNWpjbhspQalY+5uy7Ar9lJ/D39YfQVcgwpp0Tjk5uj74t7KEo7vd49FpMDXSxebgX3O1M8SgzDwPWhuHmwwypyyKiMmZrboDi/jOqEoCfu7TTxpc4FC9fvhxr167F9OnToVD8/z08mzdvjsjISK0WRySFz7u4Qk8hx8mYZBy//lDqcqgYCpQqbAm9jfZLjmJjyG0oVQKd3KwRPLEdpnZtCBMDzsJV1swMdbF1hDca2poiOSMXA9eexu1/hrEQUeWmUgksPRSNDaduF2t9GQAzQx10dbct1bpepsSh+NatW2jatOkzy/X19ZGZyQ88qvjsaxhhSCsHAMCCwChO6FHOnYpJRvfvT2Lm3it4kpWPBtbVsHWEN9YOaQ5HC2Opy6vSzI308PNIb7hYmyAxLRcD1p7G3UdZUpdFRKUoNTsfIzefw/d/xQAAOja0gkxWGHyLIvvnP9/0aSL5hEklDsV169ZFRETEM8sPHjyIhg0baqMmIsmNf8sZZoa6uJ6YgV/D70tdDhXhdnImRm0+h0E/hSE6MR3mRrqY17MRAj9+E23qW0hdHv2jhrEefh7lDWeraohPzcGAtadx/wmDMVFldD0xHT1/OIm/riVBX0eOb/t5YF1AC6wZ3BymhoX3dng6iu3pn6aGOlg7uDl83aQdOgGU4O4T//vf/zB58mRMmjQJ48aNQ05ODoQQOHPmDLZv346FCxfip59+Ks1aicqMuZEePnrLGV/uj8I3wdfh71ELxvq8dVd5kJ6Tjx+OxmDDydvIU6qgkMswuKUDJvjWh7mRtBdpUNEsqulj20hv9F9zGrHJmRiw9jR2jm6FWuaGUpdGRFoSGBmPybsuIitPCTtzQ/w42BPudmYAgE5u1gib5osDl+NxIDIesfcT4FTbBl0b26Kru63kZ4ifKva/8nPnzsXYsWMxcuRIGBoaYsaMGcjKysLAgQNRq1YtfPfdd+jfv39p1kpUpga3csDm0Du4+zgLa/6OxcRODaQuqUpTqQR+Db+PRUHRSM7IBQC8Wd8Cs952Q31rE4mro5exMjXAtlEt0W9NKO48ysLAtaexY3Qr2JhxSm2iikypElgcFI3V/0x81dq5JpYPaIYaxponKQx0FXinaW287W6NwMBAdOvWBLq65et6j2IPn/j3VfiDBg3CjRs3kJGRgYSEBNy/fx8jRowolQKJpKKvo8DnXQon9FjzdywS03IkrqjqOnv7MXquOIXPdl9CckYu6loYY11Ac2we7sVAXIHYmBUG49rVDXH7n2CclM6/V0QV1ZPMPAzdcEYdiEe3dcKmYV7PBOKKokRjiv87DaqRkRGsrKy0WhBRedKtsQ2a1jFHdr4SSw9dl7qcKicuJRsfbb+APqtDERmXChN9HUzv1hBBE9qiY0NrTs1cAdmZG2L7qJawMzdEbHImBq4NU5/5J6KK48qDVPj/cBInbiTDUFeB7wc0xbRuDaGjeKUpMMqFElXeoEED1KhR44U/RJWJTCbDjH8m9Pgl/B6i4tMkrqhqyMorwNLg6+j4zTH8cfEBZDJggJc9jk5pj1FtnaCnU3E/dKnwDi/bRnnDxtQAMUkZeP+nMDzOlHYmKyIqvr0Rcei9KgT3n2SjTg0j7PnQBz08akld1msr0ZVDc+fOhZmZWWnVQlQueTrUQLfGNgiMTMDCA9ewebiX1CVVWkII7Lv4AF8duIb41MKv1b3q1sCst93UF2xQ5eBQ0xjbR7dEvx9DcS0hHe//FIZto7x5sSRROVagVGHhgWtYd/IWAKBdA0t8179Jpfl7W6JQ3L9/fw6XoCrpMz9XBF9NxN/XH+Lv6w/RtoGl1CVVOpfup2DuH1cRfucJgMKv2ad3b4iu7jYcJlFJ1bUwxrZRLdF/zWlcjU/D4HVnsHWkN8wMy9fFN0QEJGfkYvy28zgd+xgAMK5DPUzq5FKpZgot9neQ/EeJqjJHC2O83/L/J/RQqjj9s7YkpeVg8q6L6PHDKYTfeQIjPQUmd26AI5+2Q7fGtvzsqeScraph2yhv1DTWQ2RcKgLWn0F6Tr7UZRHRv1y6n4Iey0/idOxjGOspsPr9Zpji51qpAjHwinefIKqKPn6rPkwMdHAtIR27z3NCj9eVk6/EymMx6LDkmHqClHeb2uGvT9tj/Fv1y819K6n0NbA2wdaR3jA30kXEvRQM3XAWGbkFUpdFRAB2nbuH91aH4kFqDpwsjPH7uNboIvF0zKWl2KFYpVJx6ARVadWNCyf0AIBvDkUjK4//aL8KIQQOXk5Ap2+PY9HBaGTmKdHE3hy/feiDpf2a8L61VVRDW1NsHeENUwMdhN95guEbzvLvGJGE8gpUmPn7ZUz59RLyClTwbWiF38e3rtS3weQl3EQlMKSVI2pXN0RiWi5+OnFL6nIqnKj4NAxcG4axW8Nx73E2rE318W0/D+z5wAdN61SXujySmLudGbaM8IaJvg7O3H6MERvPITtPKXVZRFVOUnoOBq49jS2n7wAAJvo2KJyq2aByj/dnKCYqAQNdBT77Z0KP1cdvcuKBYnqUkYvpv0Wi+/cnEBr7CHo6cnz0ljP++rQ93mlaG/JKNi6NXp2HvTk2jfBCNX0dhMY+wugt55CTz2BMVFbO330C/+Unce7OE5jo62BdQHN84lu/SnxOMxQTlZD/G7bwsDdHVp4S3wbfkLqcci1fqcK6k7fQfskx/Bx2FyoBdG9siyOT2uHTzi4w1i/RDXCoimhWpzo2DmsBIz0FTtxIxtit4cgtYDAmKm3bwu6i34+hSEzLRX2ratg7vjU6NrSWuqwyw1BMVEIymQzTuxVO6LHz7F1cT0yXuKLy6Wh0Eros+xvz/ryK9JwCuNmaYufollgxqBnsaxhJXR6Vc80da2D90BYw0JXjWPRDjPv5PPIKVFKXRVQp5RYoMXXPJUz7LRL5SoGu7jb4bVxrOFlWk7q0MsVQTPQKvOrWgF8ja6gEsDAwSupyypWYpAwM3XAGwzacxc2HmahprIeF7zbGHx+1gbdTTanLowqkpVNNrA9oAX0dOQ5HJeGj7eeRr2QwJtKmhNQc9PvxNLafuQeZDPisiwtWDmqGalXwmzyGYqJX9HkXV+jIZTga/RCnYpKlLkdyqdn5+N8fV9Fl2d84Fv0QugoZRrd1wtEp7THAq06lu58llQ0fZwusHdIcejpyBF1JxIQdEShgMCbSijO3HuPt5ScRcS8FZoa62DjMCx+2d66y94dnKCZ6RU6W1TDIuw4AYP7+KKiq6IQeSpXA1tN30GHJMaw/dQsFKgHfhlY4NLEdpnVrWOmvVqbS17aBJX583xN6Cjn2R8bj010XOYEO0WsQQmBTyG0MXHsayRm5cLUxwR/j26BdFZ+tlaGY6DV83LE+TPR1cDU+Db9diJO6nDIXcjMZ3b8/gRm/X8bjzDzUt6qGzcO98FNAC9S1MJa6PKpEOrhaYcWgZtCRy7A34gGm/MpgTPQqcvKV+HTXRczedwUFKoEeHrWw50Mf1KnJaz0YioleQ81q+viwQ+GEHksORVeZe6refZSFsVvCMXBtGK4lpMPMUBdz/N0Q+MmbaFvFzzRQ6enkZo3lA5pCIZdhz/k4TN1zqcp+Q0P0Ku4/ycJ7q0Ow53wcFHIZZnRviO/6N4GRXtUbP1wUvgtEr2lYa0dsPX0HcSnZWH/qFsb9E5Iro4zcAqw8GoOfTtxCnlIFhVyGQd51MNG3Aaob60ldHlUBXRvbYplK4JMdF/DLufvQUcgxv5d7lR0DSVRcITHJGL/9Ah5n5qGGsR5+GNgUPvUspC6rXGEoJnpNBroKTPFzwYSdEVh5NAZ9m9vD0kRf6rK0SqUS2HMhDl8fvIaH6bkAgDbOFpj5thtcbCrvlJ9UPvl71IJSJTDxlwhsC7sLHbkMc3s0YjAmKoIQAutO3sKCwCioBOBuZ4ofBzeHnbmh1KWVOwzFRFrQw6MW1p28hci4VHx35Dq+7NVY6pK0JvzOY/zvj6u4eD8VAOBY0wjTu7vBt6EVQwhJpldTOxSoBKb8ehGbQ+9ARy7HzLcbsk8S/UtWXgE+3x2JPy4+AAC828wOC95pDANdhcSVlU8MxURaIJfLMK1bQwxYW3ivx6E+jnC2qthnUONTs/HVgWvYG1H4YVpNXwcfveWMoa0doa/DD1SS3nuetaFUqfD57kisP3ULujoyfNHFlcGYCIXXfozecg7XEtKhI5dh5ttuGNLKgX8/XoChmEhLWtWrCd+G1jgclYivDlzDTwEtpC7plWTnKbHm71isPn4T2flKyGRAX097TPZzqXTDQqji69eiDvKVAjN+v4wfj8dCVy7Hp50b8B9+qtKOX3+Ij7dfQGp2Piyq6WHlIE941a0hdVnlHkMxkRZ90dUVR6OTcDgqCaE3H6FVvYozg5sQAn9eisdXB64hLiUbANDCsTpm+zeCu52ZxNURPd/7LR2gVAnM3ncFPxyNgY5Chgm+DaQui6jMCSGw8thNLDkUDSGAJvbmWP2+J2zMDKQurUJgKCbSImerahjgZY+tp+9iQWAU9o5rDXkFmMkt8n4q/vfnFZy9/QQAYGduiKndXNG9sS3PuFGFEODjiHylCl/uj8Kywzegq5BX6jvBEP1XRm4Bpuy6iAOXEwAAA7zsMadHIw53KwGGYiItm+DbAL9feIDIuFTsu/gAvZraSV3ScyWl52BJUDR2hd+HEIChrgIftK+H0W2deCEGVTgj33RCgUrgqwPXsDgoGjpyGca0qyd1WUSlLvZhBsZsCceNpAzoKmSY28MdA/+ZcZWKj6GYSMssqunjg/b1sDgoGouDotHF3abcBczcAiU2nLqNH/6KQUZuAQCgV5Na+LyrK2zNeJseqrjGtquH/AIVvgm+joUHrkEhl2Hkm05Sl0VUao5EJWLCjgik5xbA2lQfKwd5wtOhutRlVUgMxUSlYHjruuoJPTacuo0P2pePs1VCCARfTcT8wCjceZQFAPCobYZZ/o34IUqVxkcd6yNfJfD9kRv4cn8UdBVyBPg4Sl0WkVapVALf/3UDyw7fAFB4DciKQc1gZcLxw6+KoZioFBjqKTC5sws+3XXxnwk9aqNmNWnv3BCdkI55f17FyZhkAICViT4+7+KKd5raVYhxz0QlMdG3PgqUKqw8dhOz912BjkKGQd4OUpdFpBVpOfmYtDMCh6OSAABDWjlgRnc36OnIJa6sYmMoJiol7zS1w7qTt3A1Pg3fH7mBuT3dJanjSWYevj18HVtP34FKAHo6cox6sy4+bO8MY31+BFDlJJPJMMXPBQUqgTV/x2L6b5ehK5ejbwt7qUsjei03EtMxZks4YpMzoadTOM15n+bs19rAfxGJSolcLsOM7g0x8Kcw/Bx2F0N8HFHPslqZtZ+vVGHr6TtYdvgGUrPzAQBd3W0wrVtD2NcwKrM6iKQik8kwtasr8pUqbDh1G5/vuQSFXIbenrWlLo3olRy8HI9Pf7mIzDwlapkZYPVgT7xR21zqsioNhmKiUuTjbIG3XK3w17UkfH3gGtYMaV4m7R6//hDz/ryKmKQMAICrjQlm+zeqUPdNJtIGmUyGWW+7QakS2Bx6B1N+vQgdhQw9m5Tfu8IQ/ZdSJfDNoWisPHYTANDSqQZWDGwm+bC8ykbywScrVqyAo6MjDAwM4O3tjTNnzrxw/WXLlsHFxQWGhoawt7fHxIkTkZOTo34+PT0dEyZMgIODAwwNDeHj44OzZ88+s52oqCj06NEDZmZmMDY2RosWLXD37l2t7x/R1K6ukMuAQ1cTcebW41JtK/ZhBkZsPIuA9WcQk5SBGsZ6mP+OO/Z//CYDMVVZMpkMc/wbYYBXHagEMHFnBPZfipe6LKJiScnKw7CNZ9WBeGSbutg6wpuBuBRIGop37tyJSZMmYfbs2Th//jw8PDzg5+eHpKSkItfftm0bvvjiC8yePRtRUVFYt24ddu7ciWnTpqnXGTlyJIKDg7FlyxZERkaic+fO8PX1RVxcnHqdmzdvok2bNnB1dcWxY8dw6dIlzJw5EwYGvGKTtK++tQn6tSi8X+T8/VehUgmtt5GWk4/5+6/Cb9nfOHItCTpyGUa0qYujk9tjkLcDFLyQjqo4uVyG+b3c0bd5bagE8PGOCzj4zyQHROVVVHwaevxwCn9ffwgDXTm+698EM952g45C8nOalZKk7+rSpUsxatQoDBs2DG5ubli9ejWMjIywfv36ItcPCQlB69atMXDgQDg6OqJz584YMGCA+uxydnY2du/ejUWLFqFt27ZwdnbGnDlz4OzsjFWrVqm3M336dHTr1g2LFi1C06ZNUa9ePfTo0QNWVlZlst9U9UzsVB9GegpcvJ+KPyO1d4ZKqRLYfuYuOiw+hrUnbiFfKdDBxRJBE9ti5ttuMDPU1VpbRBWdXC7DwnffwLtN7aBUCXy0/TwOX02UuiyiIv1x8QHeXRmCu4+zYF/DEHs+aM1hP6VMsjHFeXl5CA8Px9SpU9XL5HI5fH19ERoaWuRrfHx8sHXrVpw5cwZeXl6IjY1FYGAgBg8eDAAoKCiAUql85oyvoaEhTp48CQBQqVTYv38/PvvsM/j5+eHChQuoW7cupk6dil69ej233tzcXOTm5qofp6WlAQDy8/ORn5//Su9BSTxtoyzaIu2rbqDAqDaO+O6vm/j6QBTeql8D+v9M6PGqxzbs1mPMD4xGVEI6AMDJwhjTu7mgbX2LV9oeVQ78rHi5Bb3ckFugxP7IBHzwczhWDWyCdg0spS6r3GKfKlsFShWWBN/AulN3AABtnGvi2z5vwNxIt1Icg7LuTyVpRyaE0P53ucXw4MED2NnZISQkBK1atVIv/+yzz3D8+HGEhYUV+brvv/8ekydPhhACBQUFGDt2rMZZYB8fH+jp6WHbtm2wtrbG9u3bERAQAGdnZ0RHRyMhIQG2trYwMjLCl19+iQ4dOuDgwYOYNm0ajh49inbt2hXZ7pw5czB37txnlm/btg1GRrySn14uVwnMv6BAar4MPR2UeKvWq/3Ve5QD7LsjR8Tjwi96DBUCXexVeNNagN+oERWPUgCbrxf+PdKRCYxyVcHVXJJ/DonUMvKBTTfkuJ5a+GHuW0uF7nVU4Ai4V5eVlYWBAwciNTUVpqamL1y3QoXiY8eOoX///vjyyy/h7e2NmJgYfPLJJxg1ahRmzpwJoHC88PDhw/H3339DoVCgWbNmaNCgAcLDwxEVFaVud8CAAdi2bZt62z169ICxsTG2b99eZL1FnSm2t7dHcnLyS99kbcjPz0dwcDA6deoEXV1+JV5R/Xo+DlN/uwITAx0cmdgG1Y30in1sM3ML8OOJW1h36g7yCgo/JPu3qI1P3nJGDWO9MtwLKs/4WVF8+UoVPtl5CcFRSdDXkWPt4KZo5cQLUv+LfapsXHmQhnHbIxCXkgMjPQW+eqcRurrbSF2W1pV1f0pLS4OFhUWxQrFkwycsLCygUCiQmKg5nisxMRE2NkV3gpkzZ2Lw4MEYOXIkAKBx48bIzMzE6NGjMX36dMjlctSrVw/Hjx9HZmYm0tLSYGtri379+sHJyUndro6ODtzc3DS23bBhQ/UQi6Lo6+tDX//ZKz11dXXL9EOirNsj7erbwgGbQu/iWkI6Vv19G7P9G6mfe96xVakEfo+Iw9cHryExrfAXM596NTHL3w2uNqX/CxlVTPyseDldXWDFIE98sDUcR64lYczWCGwc1gLeDMZFYp8qPXvO38fUPZHILVChroUxfhzsiQbWJlKXVarKqj+VpA3JvmzV09ODp6cnjhw5ol6mUqlw5MgRjTPH/5aVlQW5XLNkhaJwXOZ/T3gbGxvD1tYWT548QVBQEHr27Klut0WLFoiOjtZY//r163Bw4BSgVLoUchmmd28IANgcchs/Hr+JcdsjsPyKHOO2R2DP+fvIyVeq179w9wneXRWCSb9cRGJaLurUMMKPgz3x80hvBmIiLdDTkWPl+83QroElsvOVGLbxLMLvlO6tE4meyleqMGffFUz65SJyC1R4y9UKv49rXekDcXkl6eQdkyZNQkBAAJo3bw4vLy8sW7YMmZmZGDZsGABgyJAhsLOzw8KFCwEA/v7+WLp0KZo2baoePjFz5kz4+/urw3FQUBCEEHBxcUFMTAymTJkCV1dX9TYBYMqUKejXrx/atm2rHlP8xx9/4NixY2X+HlDV82Z9SzSqZYorD9Kw8MA1yGWASsgRG5WEQ1eTMOePK5jZ3Q0hNx/htwuFtxI01lNg/Fv1MbyNI/R1FBLvAVHloq+jwI+DPTFy0zmcjElGwPqz2DLCC03rVJe6NKrEHqbnYty28+r713/csT4mdKwPOQcQS0bSUNyvXz88fPgQs2bNQkJCApo0aYKDBw/C2toaAHD37l2NM8MzZsyATCbDjBkzEBcXB0tLS/j7+2P+/PnqdVJTUzF16lTcv38fNWrUQO/evTF//nyN0+fvvPMOVq9ejYULF+Ljjz+Gi4sLdu/ejTZt2pTdzlOVFXw1EVcfpKkfP71t8dM/07ILMOXXSwAAmQx4r1ltTPFzgZUp76NNVFoMdBVYO6Q5hm88i9DYRxiy/gx+HunNKXSpVETcS8HYLeFISMtBNX0dLO3rgc6NKt/44YpGsgvtKrq0tDSYmZkVa+C2NuTn5yMwMBDdunXjmK4KLCdfCa8Fh5GeXYCX/cVTyGXYObolmjvWKJPaqHLgZ8XrycorwND1Z3Hm9mOYGeri55HecLczk7osSbFPadfOs3cx8/cryFOqUM/SGD8Obg5nq2pSl1Vmyro/lSSv8QZORGUoMDIeacUIxEDhxBz3nmSVek1E9P+M9HSwflgLNKtjjtTsfAxeF4ZrCWkvfyHRS+QWKDHtt0h8vjsSeUoVOrtZ4/dxratUIC7vGIqJytChK4nFvt+kXAYEXeZsW0RlrZq+DjYO94KHvTmeZOVj0Now3EhMl7osqsAS03IwYM1pbAu7C5kMmNy5AVa/7wkTA555L08YionKUEpWnnrs8MuoBJCSnVe6BRFRkUwNdLF5uBfc7UzxKDMPA9aGISYpQ+qyqAI6d/sx3l5+EufvpsDUQAfrh7bA+Ld4QV15xFBMVIbMjfRKdKbY3JCTchBJxcxQF1tHeKOhrSmSM3IxcO1p3ErOlLosqiCEENgSehv915zGw/RcuFibYN/4NujgYiV1afQcDMVEZahzI+sSnSn2c7cu3YKI6IXMjfTw80hvuFibICm9MBjffcSx/vRiOflKfPbrJczcewUFKoHub9hiz4c+cLQwlro0egGGYqIy1K2xLUwNdfCyk8UyAGaGOujqblsWZRHRC9Qw1sPPo7zhbFUN8ak5GLD2NO7zIlh6jgcp2ej7Yyh2hd+HXAZM7eqKHwY0hbG+pHfBpWJgKCYqQwa6Cizt0wSQ4bnBWPbPf77p0wQGupyog6g8sKimj20jveFkYYy4lGwMWHsaD1KypS6LypnQm4/gv/wkLt1PhbmRLjYP98aYdvUgk3H8cEXAUExUxnzdrLFmcHOYGhaeNXg6xvjpn6aGOlg7uDl83Th0gqg8sTI1wLZRLeFQ0wj3Hmdj4NrTSEjNkbosKgeEEFh38hbeXxeGR5l5cLM1xR/j26BNfQupS6MS4Ll8Igl0crNG2DRfHLgcjwOR8Yi9nwCn2jbo2tgWXd1teYaYqJyyMTPA9lEt0W9NKG4/ysLAtaexY0xLWJlwxsmqKjtPiS/2XMLeiAcAgHea2mHBO41hqMfP8YqGZ4qJJGKgq8A7TWtjxYAm+KiRCisGNME7TWszEBOVc7XMDbFtZEvYmRsiNjkTA9eGITkjV+qySAL3Hmeh96oQ7I14AIVchtn+blja14OBuIJiKCYiIioh+xpG2DbKGzamBohJysD7P4XhcSbvK16VnLjxEP4/nMTV+DTUNC68S8mw1nU5frgCYygmIiJ6BQ41jbF9dEtYmejjWkI63v8pDClZDMaVnRACq4/fRMD6M0jJyodHbTP88VEbtHSqKXVp9JoYiomIiF5RXQtjbBvVEhbV9HE1Pg2D151Bana+1GVRKcnMLcD4bRfw1YFrUAmgb/Pa2DmmFWqZG0pdGmkBQzEREdFrcLaqhm2jvFHTWA+RcakIWH8G6TkMxpXN7eRMvLsyBPsj46GrkOHLXu74uvcbvA6kEmEoJiIiek0NrE2wdaQ3zI10EXEvBUM3nEVGboHUZZGWHL2WhB4/nER0YjosTfSxY3RLvN/SgeOHKxmGYiIiIi1oaGuKrSO8YWqgg/A7TzB8w1lk5TEYV2QqlcDyIzcwfNNZpOUUwNOhOvZ/1AaeDjWkLo1KAUMxERGRlrjbmWHrSG+YGOjgzO3HGLHxHLLzlFKXRa8gPScfY7eG45vg6xACeL9lHWwf1RJWprwndWXFUExERKRFb9Q2x+bhXqimr4PQ2EcYveUccvIZjCuSmKQM9FpxCoeuJkJPIcfXvRvjy16NoafD2FSZ8egSERFpWdM61bFxWAsY6Slw4kYyxmwJR24Bg3FFcOhKAnqtOIWbDzNhY2qAX8a2Qr8WdaQui8oAQzEREVEpaO5YA+uHtoCBrhzHrz/Eh1vPI69AJXVZ9BwqlcDSQ9EYvSUcGbkF8KpbA3981AZN7M2lLo3KCEMxERFRKWnpVBPrA1pAX0eOI9eS8NH288hXMhiXN6nZ+Rix6Sy+/ysGADCstSN+HukNSxN9iSujssRQTEREVIp8nC2wdkhz6OnIEXQlERN2RKCAwbjciE5IR88fTuJo9EPo68jxbT8PzPZvBF0FI1JVwyNORERUyto2sMSP73tCTyHH/sh4fLrrIpQqIXVZVd7+S/F4Z+Up3H6UBTtzQ+z+wAfvNK0tdVkkEYZiIiKiMtDB1QorBzWDjlyGvREPMOVXBmOpKFUCCw9EYdy288jKU6K1c0388VEbuNuZSV0aSYihmIiIqIz4ulnjh4FNoZDLsOd8HKbuuQQVg3GZepKZh6EbzuDH47EAgDFtnbBpmBdqGOtJXBlJjaGYiIioDHVxt8V3/ZtALgN+OXcfM/ZehhAMxmXhyoNU+P9wEiduJMNQV4HlA5piareG0OH4YQKgI3UBREREVc3bb9SCUiUwYWcEtoXdhY5chrk9GkEmk0ldWqX1+4U4fLHnEnLyVahTwwhrhnjC1cZU6rKoHGEoJiIikkDPJnbIVwpM+fUiNofegY5cjplvN2Qw1rJ8pQoLA69h/albAID2Lpb4rl9TmBnpSlwZlTcMxURERBJ5z7M2lCoVPt8difWnbkFXIcMXXV0ZjLUkOSMX434+j7BbjwEA4zs4Y2KnBlDI+f7SsxiKiYiIJNSvRR3kKwVm/H4ZP/4dC12FHJ92bsBg/Jou3kvB2K3hiE/NgbGeAt/0bYIu7jZSl0XlGEMxERGRxN5v6QClSmD2viv44WgMdBQyTPBtIHVZFdYv5+5hxu+XkVeggpOlMdYM9oSzlYnUZVE5x1BMRERUDgT4OCJfqcKX+6Ow7PAN6CrkGNfBWeqyKpS8AhXm/XkVW07fAQD4NrTG0n4eMDXg+GF6OYZiIiKicmLkm04oUAl8deAaFgdFQ0cuw5h29aQuq0JISsvBhz+fx7k7TyCTARN9G2B8B2fIOX6YiomhmIiIqBwZ264eCpQqLDl0HQsPXINCLsPIN52kLqtcC7/zBB9sDUdSei5MDHTwXf8meMvVWuqyqIJhKCYiIipnxr9VH/lKge+O3MCX+6Ogq5AjwMdR6rLKpZ/D7mDOvivIVwrUt6qGNUOao66FsdRlUQXEUExERFQOTfCtjwKVCiuO3sTsfVego5BhkLeD1GWVG7kFSszeewU7zt4DAHRrbIPF73nAWJ/Rhl4New4REVE5JJPJMLmzC/KVAmv+jsX03y5DRy5DvxZ1pC5NcvGp2fhg63lE3EuBTAZ85ueKse2ceBs7ei0MxUREROWUTCbD1K6uyFeqsOHUbXyxJxI6cjl6e9aWujTJhMU+wrht55GckQczQ118P6Ap2jWwlLosqgQYiomIiMoxmUyGWW+7QakS2Bx6B1N+vQgdhQw9m9hJXVqZEkJgU8htfLk/CgUqAVcbE6wZ3Bx1ahpJXRpVEgzFRERE5ZxMJsMc/0bIVwpsP3MXE3dGQEcuR/c3bKUurUzk5Csx7bdI7DkfBwDo4VELX/VuDCM9xhjSHvYmIiKiCkAul2F+L3coVSr8cu4+Pt5xAQq5rNJPXXz/SRbGbg3H5bg0KOSFw0lGtKnL8cOkdXKpCyAiIqLikctlWPjuG3i3qR2UKoGPtp/H4auJUpdVakJikuG//CQux6WhhrEetozwwsg3eUEdlQ6GYiIiogpEIZdhcR8P9PCohXylwIc/n8fR6CSpy9IqIQTW/h2L99eF4UlWPhrbmeGPj9rAp56F1KVRJcZQTEREVMEo5DIs7euB7o1tkadUYcyWcPx9/aHUZWlFVl4BPt4RgfmBUVAJoHez2tg1thXszA2lLo0qOYZiIiKiCkhHIcey/k3Q2c0aeQUqjNp8DiExyVKX9VruPMrEuytD8MfFB9CRy/C/no2wpM8bMNBVSF0aVQHlIhSvWLECjo6OMDAwgLe3N86cOfPC9ZctWwYXFxcYGhrC3t4eEydORE5Ojvr59PR0TJgwAQ4ODjA0NISPjw/Onj2rsY2hQ4dCJpNp/HTp0qVU9o+IiKg06Crk+GFgM3R0tUJugQojNp1DWOwjqct6Jceik+C//CSuJaTDopo+to1qiSGtHDl+mMqM5KF4586dmDRpEmbPno3z58/Dw8MDfn5+SEoqenzUtm3b8MUXX2D27NmIiorCunXrsHPnTkybNk29zsiRIxEcHIwtW7YgMjISnTt3hq+vL+Li4jS21aVLF8THx6t/tm/fXqr7SkREpG16OnKsfL8Z2jWwRHa+EsM2nkX4ncdSl1VsQgisOBqDYRvPIi2nAE3szfHnR23gVbeG1KVRFSN5KF66dClGjRqFYcOGwc3NDatXr4aRkRHWr19f5PohISFo3bo1Bg4cCEdHR3Tu3BkDBgxQn13Ozs7G7t27sWjRIrRt2xbOzs6YM2cOnJ2dsWrVKo1t6evrw8bGRv1TvXr1Ut9fIiIibdPXUeDHwZ5o42yBrDwlAtafxYW7T6Qu66UycgvwwdbzWBwUDSGAAV51sHNMS9iYGUhdGlVBkt6nOC8vD+Hh4Zg6dap6mVwuh6+vL0JDQ4t8jY+PD7Zu3YozZ87Ay8sLsbGxCAwMxODBgwEABQUFUCqVMDDQ/AtlaGiIkydPaiw7duwYrKysUL16dbz11lv48ssvUbNmzSLbzc3NRW5urvpxWloaACA/Px/5+fkl3/kSetpGWbRFZYvHlrSJ/anqUgBYOcADo7eex+lbTzBk/RlsGuqJxnZmr7Xd0upTsQ8z8eH2CNx8mAldhQyz326Ifs1rA0KF/HyVVtui8qOsP6NK0o5MCCFKsZYXevDgAezs7BASEoJWrVqpl3/22Wc4fvw4wsLCinzd999/j8mTJ0MIgYKCAowdO1bjLLCPjw/09PSwbds2WFtbY/v27QgICICzszOio6MBADt27ICRkRHq1q2LmzdvYtq0aahWrRpCQ0OhUDw7oH/OnDmYO3fuM8u3bdsGIyNOMUlEROVDrhL4MUqBm+kyGCkExjVSorax1FVpuvxYhi0xcuQoZTDTFRjuooSjidRVUWWUlZWFgQMHIjU1Faampi9ct8KF4mPHjqF///748ssv4e3tjZiYGHzyyScYNWoUZs6cCQC4efMmhg8fjr///hsKhQLNmjVDgwYNEB4ejqioqCJriY2NRb169XD48GF07NjxmeeLOlNsb2+P5OTkl77J2pCfn4/g4GB06tQJurq6pd4elR0eW9Im9icCCocljNh8HufvpqC6kS62DGsOF5tXS53a7FMqlcAPx25i+dFYAEBzB3N8388Dlib6r7VdqjjK+jMqLS0NFhYWxQrFkg6fsLCwgEKhQGKi5mw8iYmJsLEpetrKmTNnYvDgwRg5ciQAoHHjxsjMzMTo0aMxffp0yOVy1KtXD8ePH0dmZibS0tJga2uLfv36wcnJ6bm1ODk5wcLCAjExMUWGYn19fejrP/uXVldXt0z/4Snr9qjs8NiSNrE/VW3VdXWxabgX3l93BhfvpSBgYzh2jG6J+tavfjr2dftUWk4+Ju64iCPXCi+kD2jlgOnd3aCnI/nlTSSBsvqMKkkbkvZEPT09eHp64siRI+plKpUKR44c0Thz/G9ZWVmQyzXLfjrc4b8nvY2NjWFra4snT54gKCgIPXv2fG4t9+/fx6NHj2Bra/uqu0NERFRumBjoYvNwLzS2M8OjzDwMWBuGmKQMSWq5kZiOnj+cwpFrSdDTkWNJHw/M7enOQEzliuS9cdKkSVi7di02bdqEqKgofPDBB8jMzMSwYcMAAEOGDNG4EM/f3x+rVq3Cjh07cOvWLQQHB2PmzJnw9/dXh+OgoCAcPHhQ/XyHDh3g6uqq3mZGRgamTJmC06dP4/bt2zhy5Ah69uwJZ2dn+Pn5lf2bQEREVArMDHWxZYQXGtqaIjkjFwPXnsat5MwyreFAZDx6rTiFW8mZsDM3xO6xPnjPs3aZ1kBUHJIOnwCAfv364eHDh5g1axYSEhLQpEkTHDx4ENbW1gCAu3fvapwZnjFjBmQyGWbMmIG4uDhYWlrC398f8+fPV6+TmpqKqVOn4v79+6hRowZ69+6N+fPnq0+hKxQKXLp0CZs2bUJKSgpq1aqFzp07Y968eUUOkSAiIqqozI308PNIbwxYcxrRiekYuPY0do5uhTo1S/cicaVK4JtD0Vh57CYAoJVTTfwwsClqVuO/s1Q+SXqhXUWWlpYGMzOzYg3c1ob8/HwEBgaiW7duHCdYyfDYkjaxP9HzJGfkov+a04hJyoCduSF2jmmJ2tVfHoxfpU+lZOXh4x0R+Pv6QwDAqDfr4vMurtBRSP4FNUmsrD+jSpLX2DuJiIiqAItq+tg20htOFsaIS8nGgLWn8SAlW+vtXH2QBv8fTuLv6w9hoCvHd/2bYHp3NwZiKvfYQ4mIiKoIK1MDbBvVEg41jXDvcTYGrj2NhNQcrW1/38UHeHfVKdx7nA37GobY80Fr9Gxip7XtE5UmhmIiIqIqxMbMANtHtYR9DUPcfpSFgWtPIyn99YJxgVKF+fuv4uPtF5CTr8Kb9S3wx/g2cKtV+sMLibSFoZiIiKiKqWVuiG0jW8LO3BCxyZkYuDYMyRm5L39hER5n5mHI+jNYe+IWAOCD9vWwcZgXzI30tFkyUaljKCYiIqqC7GsYYfuolrA1M0BMUgbe/ykMjzPzSrSNy3Gp8F9+EiE3H8FIT4GVg5rh8y6uUMhlpVQ1UelhKCYiIqqi6tQ0wrZRLWFloo9rCel4/6cwpGQVLxjvDr+P3qtCEJeSjboWxvh9XGt0a8wJsKjiYigmIiKqwupaGGPbqJawqKaPq/FpGLzuDFKz85GTr8Se8/cxbnsEll+RY9z2COw5fx8ZOfmYs+8KPt11EbkFKrzlaoXfx7VGg9eYQpqoPJB88g4iIiKSlrNVNWwbVTjBR2RcKnquOIXHGblIyymAXAaohByxUUk4dDUJCrkMSlXhFAcfd6yPCR3rQ87hElQJ8EwxERERoYG1CbaO9IaxngK3kzORllMAAPgn/6r/fBqIx3VwxqRODRiIqdJgKCYiIiIAhUMpZLKXh1wZgK2nbyMnX1n6RRGVEYZiIiIiAgAERsYjI7fgpesJAKnZBThwOb70iyIqIwzFREREBAA4dCURxR0NIZcBQZcTS7cgojLEUExEREQAgJSsPPXY4ZdRCSAlu2T3NSYqzxiKiYiICABgbqRXojPF5oactY4qD4ZiIiIiAgB0bmRdojPFfu7WpVsQURliKCYiIiIAQLfGtjA11MHLThbLAJgZ6qCrO2ewo8qDoZiIiIgAAAa6Cizt0wSQ4bnBWPbPf77p0wQGuoqyK46olDEUExERkZqvmzXWDG4OU8PCSW+fjjF++qepoQ7WDm4OXzcOnaDKhdM8ExERkYZObtYIm+aLA5fjcSAyHrH3E+BU2wZdG9uiq7stzxBTpcRQTERERM8w0FXgnaa18ba7NQIDA9GtWxPo6upKXRZRqeHwCSIiIiKq8hiKiYiIiKjKYygmIiIioiqPoZiIiIiIqjyGYiIiIiKq8nj3iVckROE8mGlpaWXSXn5+PrKyspCWlsarfysZHlvSJvYn0jb2KdKmsu5PT3Pa09z2IgzFryg9PR0AYG9vL3ElRERERPQi6enpMDMze+E6MlGc6EzPUKlUePDgAUxMTCCTvWyW+NeXlpYGe3t73Lt3D6ampqXeHpUdHlvSJvYn0jb2KdKmsu5PQgikp6ejVq1akMtfPGqYZ4pfkVwuR+3atcu8XVNTU34oVVI8tqRN7E+kbexTpE1l2Z9edob4KV5oR0RERERVHkMxEREREVV5DMUVhL6+PmbPng19fX2pSyEt47ElbWJ/Im1jnyJtKs/9iRfaEREREVGVxzPFRERERFTlMRQTERERUZXHUExEREREVR5DMRERERFVeQzFpWTFihVwdHSEgYEBvL29cebMmReuv2vXLri6usLAwACNGzdGYGCgxvNCCMyaNQu2trYwNDSEr68vbty4obHO48ePMWjQIJiamsLc3BwjRoxARkaG+vmcnBwMHToUjRs3ho6ODnr16qW1/a1KyuOxvX37NmQy2TM/p0+f1t6OU6mQoj/Nnz8fPj4+MDIygrm5eZHt3L17F927d4eRkRGsrKwwZcoUFBQUvNa+Uukrr/2pqM+nHTt2vNa+Utko6z51+/ZtjBgxAnXr1oWhoSHq1auH2bNnIy8vT2M7ly5dwptvvgkDAwPY29tj0aJFr7+zgrRux44dQk9PT6xfv15cuXJFjBo1Spibm4vExMQi1z916pRQKBRi0aJF4urVq2LGjBlCV1dXREZGqtf56quvhJmZmfj999/FxYsXRY8ePUTdunVFdna2ep0uXboIDw8Pcfr0aXHixAnh7OwsBgwYoH4+IyNDjB07VqxZs0b4+fmJnj17ltp7UFmV12N769YtAUAcPnxYxMfHq3/y8vJK782g1yZVf5o1a5ZYunSpmDRpkjAzM3umnYKCAuHu7i58fX3FhQsXRGBgoLCwsBBTp07V+ntA2lNe+5MQQgAQGzZs0Ph8+vc2qHySok8dOHBADB06VAQFBYmbN2+KvXv3CisrK/Hpp5+qt5Gamiqsra3FoEGDxOXLl8X27duFoaGh+PHHH19rfxmKS4GXl5cYN26c+rFSqRS1atUSCxcuLHL9vn37iu7du2ss8/b2FmPGjBFCCKFSqYSNjY1YvHix+vmUlBShr68vtm/fLoQQ4urVqwKAOHv2rHqdAwcOCJlMJuLi4p5pMyAggKH4FZTXY/s0FF+4cEEr+0llQ4r+9G8bNmwoMsQEBgYKuVwuEhIS1MtWrVolTE1NRW5ubon2kcpOee1PQhSG4t9++62Ee0RSk7pPPbVo0SJRt25d9eOVK1eK6tWra3weff7558LFxaVkO/gfHD6hZXl5eQgPD4evr696mVwuh6+vL0JDQ4t8TWhoqMb6AODn56de/9atW0hISNBYx8zMDN7e3up1QkNDYW5ujubNm6vX8fX1hVwuR1hYmNb2ryqrCMe2R48esLKyQps2bbBv377X22EqVVL1p+IIDQ1F48aNYW1trdFOWloarly5UuztUNkpz/3pqXHjxsHCwgJeXl5Yv349BKdJKNfKU59KTU1FjRo1NNpp27Yt9PT0NNqJjo7GkydPSraj/8JQrGXJyclQKpUa/5gAgLW1NRISEop8TUJCwgvXf/rny9axsrLSeF5HRwc1atR4brtUMuX52FarVg3ffPMNdu3ahf3796NNmzbo1asXg3E5JlV/Ko7ntfPvNqh8Kc/9CQD+97//4ZdffkFwcDB69+6NDz/8EMuXLy/RNqhslZc+FRMTg+XLl2PMmDEvbeffbbwKnVd+JRGVGxYWFpg0aZL6cYsWLfDgwQMsXrwYPXr0kLAyIiJg5syZ6v9v2rQpMjMzsXjxYnz88ccSVkXlXVxcHLp06YI+ffpg1KhRpd4ezxRrmYWFBRQKBRITEzWWJyYmwsbGpsjX2NjYvHD9p3++bJ2kpCSN5wsKCvD48ePntkslU9GOrbe3N2JiYoqxZyQFqfpTcTyvnX+3QeVLee5PRfH29sb9+/eRm5v7Wtuh0iN1n3rw4AE6dOgAHx8frFmzpljt/LuNV8FQrGV6enrw9PTEkSNH1MtUKhWOHDmCVq1aFfmaVq1aaawPAMHBwer169atCxsbG4110tLSEBYWpl6nVatWSElJQXh4uHqdv/76CyqVCt7e3lrbv6qsoh3biIgI2NralnxHqUxI1Z+Ko1WrVoiMjNT4ZSw4OBimpqZwc3Mr9nao7JTn/lSUiIgIVK9eHfr6+q+1HSo9UvapuLg4tG/fHp6entiwYQPkcs242qpVK/z999/Iz8/XaMfFxQXVq1d/9Z1+rcv0qEg7duwQ+vr6YuPGjeLq1ati9OjRwtzcXH0l9+DBg8UXX3yhXv/UqVNCR0dHLFmyRERFRYnZs2cXeQsTc3NzsXfvXnHp0iXRs2fPIm/b1bRpUxEWFiZOnjwp6tevr3HbLiGEuHLlirhw4YLw9/cX7du3FxcuXOAdC0qgvB7bjRs3im3btomoqCgRFRUl5s+fL+RyuVi/fn0ZvCv0qqTqT3fu3BEXLlwQc+fOFdWqVVN/DqSnpwsh/v+WbJ07dxYRERHi4MGDwtLSkrdkK+fKa3/at2+fWLt2rYiMjBQ3btwQK1euFEZGRmLWrFll9M7Qq5KiT92/f184OzuLjh07ivv372vcxu+plJQUYW1tLQYPHiwuX74sduzYIYyMjHhLtvJq+fLlok6dOkJPT094eXmJ06dPq59r166dCAgI0Fj/l19+EQ0aNBB6enqiUaNGYv/+/RrPq1QqMXPmTGFtbS309fVFx44dRXR0tMY6jx49EgMGDBDVqlUTpqamYtiwYeoPpaccHBwEgGd+qPjK47HduHGjaNiwoTAyMhKmpqbCy8tL7Nq1S/s7T1onRX8KCAgo8nPg6NGj6nVu374tunbtKgwNDYWFhYX49NNPRX5+vtb3n7SrPPanAwcOiCZNmohq1aoJY2Nj4eHhIVavXi2USmWpvAekXWXdpzZs2FBkf/pvVrl48aJo06aN0NfXF3Z2duKrr7567X2VCcF7ohARERFR1cYxxURERERU5TEUExEREVGVx1BMRERERFUeQzERERERVXkMxURERERU5TEUExEREVGVx1BMRERERFUeQzERERERVXkMxUREZWjo0KHo1avXa29n48aNMDc3f+3tvIxMJsPvv/9e6u2UhpLWfuzYMchkMqSkpJRaTURUfjEUE1GlN3ToUMhkMshkMujq6qJu3br47LPPkJOTI3Vpr6xfv364fv261rY3Z84cNGnS5Jnl8fHx6Nq1q9ba+a9r165BJpPh9OnTGstbtmwJAwMDjWOUk5MDAwMDrFu3rljbLo3an/c+EVHFx1BMRFVCly5dEB8fj9jYWHz77bf48ccfMXv2bKnLeiX5+fkwNDSElZVVqbdlY2MDfX39Utu+q6srbGxscOzYMfWy9PR0nD9/HpaWlhphOTQ0FLm5uXjrrbeKte3Srp2IKheGYiKqEvT19WFjYwN7e3v06tULvr6+CA4OVj+vUqmwcOFC1K1bF4aGhvDw8MCvv/6qsY19+/ahfv36MDAwQIcOHbBp0yaNr9uLOou4bNkyODo6PreugwcPok2bNjA3N0fNmjXx9ttv4+bNm+rnb9++DZlMhp07d6Jdu3YwMDDAzz///MzwCUdHR/XZ8H//PPX555+jQYMGMDIygpOTE2bOnIn8/HwAhUMx5s6di4sXL6pft3HjRgDPDkGIjIzEW2+9BUNDQ9SsWROjR49GRkaG+vmnw0OWLFkCW1tb1KxZE+PGjVO3VZQOHTpohOKTJ0+iQYMG8Pf311h+7NgxODg4oG7dugCAvXv3olmzZjAwMICTkxPmzp2LgoIC9fr/rT0kJARNmjSBgYEBmjdvjt9//x0ymQwREREa9YSHh6N58+YwMjKCj48PoqOjX/o+EVHFx1BMRFXO5cuXERISAj09PfWyhQsXYvPmzVi9ejWuXLmCiRMn4v3338fx48cBALdu3cJ7772HXr164eLFixgzZgymT5/+2rVkZmZi0qRJOHfuHI4cOQK5XI533nkHKpVKY70vvvgCn3zyCaKiouDn5/fMds6ePYv4+HjEx8fj/v37aNmyJd5880318yYmJti4cSOuXr2K7777DmvXrsW3334LoHAoxqeffopGjRqpt9GvX78ia/Xz80P16tVx9uxZ7Nq1C4cPH8b48eM11jt69Chu3ryJo0ePYtOmTdi4ceMLw2OHDh1w8uRJdaA9evQo2rdvj3bt2uHo0aMa2+3QoQMA4MSJExgyZAg++eQTXL16FT/++CM2btyI+fPnF9lGWloa/P390bhxY5w/fx7z5s3D559/XuS606dPxzfffINz585BR0cHw4cPL9H7REQVlCAiquQCAgKEQqEQxsbGQl9fXwAQcrlc/Prrr0IIIXJycoSRkZEICQnReN2IESPEgAEDhBBCfP7558Ld3V3j+enTpwsA4smTJ0IIIWbPni08PDw01vn222+Fg4ODRi09e/Z8bq0PHz4UAERkZKQQQohbt24JAGLZsmUa623YsEGYmZkVuY2PP/5YODg4iKSkpOe2s3jxYuHp6al+XFTtQggBQPz2229CCCHWrFkjqlevLjIyMtTP79+/X8jlcpGQkKDePwcHB1FQUKBep0+fPqJfv37PreXGjRsCgPr9b9Gihfjll1/EgwcPhL6+vsjOzhZZWVlCX19fbNq0SQghRMeOHcWCBQs0trNlyxZha2tbZO2rVq0SNWvWFNnZ2ern165dKwCICxcuCCGEOHr0qAAgDh8+rLF/ANSve977REQVn45UYZyIqCx16NABq1atQmZmJr799lvo6Oigd+/eAICYmBhkZWWhU6dOGq/Jy8tD06ZNAQDR0dFo0aKFxvNeXl6vXdeNGzcwa9YshIWFITk5WX2G+O7du3B3d1ev17x582Jtb82aNVi3bh1CQkJgaWmpXr5z5058//33uHnzJjIyMlBQUABTU9MS1RoVFQUPDw8YGxurl7Vu3RoqlQrR0dGwtrYGADRq1AgKhUK9jq2tLSIjI5+7XWdnZ9SuXRvHjh1Do0aNcOHCBbRr1w5WVlaoU6cOQkNDIYRAbm6u+kzxxYsXcerUKY0zw0qlEjk5OcjKyoKRkZFGG9HR0XjjjTdgYGCgXva84/fGG29o1A4ASUlJqFOnzkvfIyKquBiKiahKMDY2hrOzMwBg/fr18PDwwLp16zBixAj1mNj9+/fDzs5O43UluVBLLpdDCKGx7EVjaQHA398fDg4OWLt2LWrVqgWVSgV3d3fk5eU9U//LHD16FB999BG2b9+uEexCQ0MxaNAgzJ07F35+fjAzM8OOHTvwzTffFHvfSkJXV1fjsUwme2Y4yH+1b98eR48exRtvvIH69eurLyJ8OoRCCAFnZ2fY29sDADIyMjB37ly8++67z2zr38H3det/Oi77ZfUTUcXHUExEVY5cLse0adMwadIkDBw4EG5ubtDX18fdu3fRrl27Il/j4uKCwMBAjWVnz57VeGxpaYmEhAQIIdRh6r8Xcf3bo0ePEB0djbVr16rH/548efKV9ikmJgbvvfcepk2b9kxQDAkJgYODg8YY6Dt37miso6enB6VS+cI2GjZsiI0bNyIzM1Md0k+dOgW5XA4XF5dXqvupDh064OOPP4abmxvat2+vXt62bVusXbsWQgj1WWIAaNasGaKjo9W/6LyMi4sLtm7ditzcXPUvOv89fsVRnPeJiComXmhHRFVSnz59oFAosGLFCpiYmGDy5MmYOHEiNm3ahJs3b+L8+fNYvnw5Nm3aBAAYM2YMrl27hs8//xzXr1/HL7/8onGHBqDwbOfDhw+xaNEi3Lx5EytWrMCBAweeW0P16tVRs2ZNrFmzBjExMfjrr78wadKkEu9LdnY2/P390bRpU4wePRoJCQnqHwCoX78+7t69ix07duDmzZv4/vvv8dtvv2lsw9HREbdu3UJERASSk5ORm5v7TDuDBg2CgYEBAgICcPnyZfWZ6cGDB6uHTryqDh06IDMzE+vXr9f4xaRdu3YICwvDmTNnNELxrFmzsHnzZsydOxdXrlxBVFQUduzYgRkzZhS5/YEDB0KlUmH06NGIiopCUFAQlixZAgAad+l4meK8T0RUMTEUE1GVpKOjg/Hjx2PRokXIzMzEvHnzMHPmTCxcuBANGzZEly5dsH//fvXtv+rWrYtff/0Ve/bswRtvvIFVq1apz7w+PfPYsGFDrFy5EitWrICHhwfOnDmDyZMnP7cGuVyOHTt2IDw8HO7u7pg4cSIWL15c4n1JTEzEtWvXcOTIEdSqVQu2trbqHwDo0aMHJk6ciPHjx6NJkyYICQnBzJkzNbbRu3dvdOnSBR06dIClpSW2b9/+TDtGRkYICgrC48eP0aJFC7z33nvo2LEjfvjhhxLX/F9169aFg4MD0tPTNUJxnTp1UKtWLeTl5WmcQfbz88Off/6JQ4cOoUWLFmjZsiW+/fZbODg4FLl9U1NT/PHHH4iIiECTJk0wffp0zJo1C0DJhlsU530ioopJJv47AI6IiIpl/vz5WL16Ne7duyd1KfQKfv75ZwwbNgypqakwNDSUuhwikhjHFBMRFdPKlSvRokUL1KxZE6dOncLixYufuUcvlV+bN2+Gk5MT7OzscPHiRXz++efo27cvAzERAWAoJiIqths3buDLL7/E48ePUadOHXz66aeYOnWq1GVRMSUkJGDWrFlISEiAra0t+vTp89zJPoio6uHwCSIiIiKq8nihHRERERFVeQzFRERERFTlMRQTERERUZXHUExEREREVR5DMRERERFVeQzFRERERFTlMRQTERERUZXHUExEREREVd7/AXleWA40OAnSAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 800x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Assuming `Ep` is the best epoch and `best_latent` is the optimal latent dimension\n",
    "Ep = 15  # Replace with the actual best epoch from tuning\n",
    "best_latent = 5  # Replace with the actual best latent dimension from tuning\n",
    "\n",
    "# Regularization weights to test\n",
    "regularization_weights = [0.0001, 0.0005, 0.001, 0.0015, 0.002]\n",
    "\n",
    "# Fixed parameters\n",
    "lr = 0.01  # Learning rate\n",
    "\n",
    "# Placeholder for storing test RMSE values for each regularization weight\n",
    "test_rmse_list = []\n",
    "\n",
    "# Iterate over each regularization weight\n",
    "for reg in regularization_weights:\n",
    "    # Initialize the MF model with the current regularization weight and best latent dimension\n",
    "    mf_model = MF(train_mat, test_mat, latent=best_latent, lr=lr, reg=reg)\n",
    "\n",
    "    # Train the model for the best epoch count\n",
    "    _, epoch_test_RMSE_list = mf_model.train(epoch=Ep, verbose=False)\n",
    "\n",
    "    # Record the final test RMSE after training\n",
    "    test_rmse_list.append(epoch_test_RMSE_list[-1])\n",
    "\n",
    "# Plot the results\n",
    "plt.figure(figsize=(8, 4))\n",
    "plt.plot(regularization_weights, test_rmse_list, marker='o', linewidth=1.5, markersize=8)\n",
    "plt.xticks(regularization_weights)\n",
    "plt.xlabel('Regularization Weight')\n",
    "plt.ylabel('Test RMSE')\n",
    "plt.title('Tune Regularization Weight')\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "anaconda-cloud": {},
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
