# 🚀 Spaceship Titanic - ML25 Team05 Final Project

本專案為 ML25 機器學習課程期末專題，我們使用 Keras 神經網路對 Spaceship Titanic 資料進行預測建模，目標是判斷乘客是否成功傳送（Transported）。

## 👥 組員資訊

- 112703003 資訊二 黃柏淵  
- 112703028 資訊二 吳亭翰  
- 112703037 資訊二 陳凱廷  
- 112703046 資訊二 卓俊瑋  

---

## 📊 資料說明

資料來源為 Kaggle 的 [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) 競賽資料集，包含以下兩份 CSV：

- `train.csv`：包含 Transported 標籤的訓練資料
- `test.csv`：不含標籤，用於預測與競賽提交

---

## 🔍 前處理與特徵工程

### 資料合併與處理

- 合併 `train` 與 `test` 進行統一處理
- 建立 `GroupId`，依據 `PassengerId` 判定群組關係

### 缺失值補齊策略

- 同群組乘客 98% 來自同一星球，91.6% 同一艙位，88.27% 相同目的地 → 使用群組 mode 補值
- `VIP` 僅 2.28% 為 True → 缺失值預設為 False
- 類別欄位補 `'Unknown'`、數值欄位補中位數

### 特徵拆解與新增

- 拆解 `Cabin` → `Deck` / `CabinNum` / `Side`
- 建立 `TotalSpend`：五種花費欄位總和

### 特徵類型分類

- 數值型特徵：`Age`, `CabinNum`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
- 類別型特徵：`HomePlanet`, `Destination`, `Deck`, `Side`, `CryoSleep`, `VIP`

---

## 🧠 模型架構

本專案使用 Keras 建立神經網路模型：

```python
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
