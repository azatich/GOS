import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
import os
import json

class BankTransferMLAnalyzer:
    """
    Продвинутый класс для ML-анализа межбанковских переводов
    с расширенной feature engineering и множественными моделями
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def engineer_features(self):
        """Создание продвинутых признаков для ML"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        df = self.df
        
        # Временные признаки
        df['quarter'] = df['month_num'].apply(lambda x: (x-1)//3 + 1)
        df['is_year_end'] = (df['month_num'] == 12).astype(int)
        df['is_year_start'] = (df['month_num'] == 1).astype(int)
        df['is_quarter_end'] = df['month_num'].isin([3, 6, 9, 12]).astype(int)
        
        # Циклические признаки (синус/косинус для месяцев)
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        
        # Лаговые признаки (предыдущие значения)
        df['amount_lag1'] = df['amount_billion_tenge'].shift(1)
        df['amount_lag2'] = df['amount_billion_tenge'].shift(2)
        df['amount_lag3'] = df['amount_billion_tenge'].shift(3)
        df['trans_lag1'] = df['transactions_thousand'].shift(1)
        df['trans_lag2'] = df['transactions_thousand'].shift(2)
        
        # Rolling статистики
        df['amount_rolling_mean_3'] = df['amount_billion_tenge'].rolling(window=3, min_periods=1).mean()
        df['amount_rolling_std_3'] = df['amount_billion_tenge'].rolling(window=3, min_periods=1).std()
        df['trans_rolling_mean_3'] = df['transactions_thousand'].rolling(window=3, min_periods=1).mean()
        
        # Темп роста
        df['amount_growth_rate'] = df['amount_billion_tenge'].pct_change() * 100
        df['trans_growth_rate'] = df['transactions_thousand'].pct_change() * 100
        
        # Разница с предыдущим месяцем
        df['amount_diff'] = df['amount_billion_tenge'].diff()
        df['trans_diff'] = df['transactions_thousand'].diff()
        
        # Взаимодействие признаков
        df['trans_amount_ratio'] = df['transactions_thousand'] / (df['amount_billion_tenge'] + 1e-6)
        df['amount_per_trans'] = df['avg_transaction_size']
        
        # Заполнение пропусков после создания лагов
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        self.df = df
        
        print(f"\n✓ Создано признаков: {len(df.columns)}")
        print(f"✓ Размер данных: {df.shape}")
        
        # Вывод корреляции с целевой переменной
        feature_cols = [col for col in df.columns if col not in ['date', 'period', 'name_kz', 'month_raw', 'month', 'id']]
        correlations = df[feature_cols].corr()['amount_billion_tenge'].sort_values(ascending=False)
        
        print("\nТоп-10 признаков по корреляции с объёмом переводов:")
        print(correlations.head(10))
        
        return df
    
    def prepare_data(self, target='amount_billion_tenge', test_size=0.2):
        """Подготовка данных для обучения"""
        print("\n" + "="*60)
        print("ПОДГОТОВКА ДАННЫХ")
        print("="*60)
        
        # Выбор признаков
        exclude_cols = ['date', 'period', 'name_kz', 'month_raw', 'month', 'id', 
                       'amount_billion_tenge', 'amount_million_tenge']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols].values
        y = self.df[target].values
        
        print(f"\nПризнаков для обучения: {X.shape[1]}")
        print(f"Примеров: {X.shape[0]}")
        
        # Разделение с учётом временной последовательности
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]
        
        # Нормализация данных
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nОбучающая выборка: {X_train.shape[0]} примеров")
        print(f"Тестовая выборка: {X_test.shape[0]} примеров")
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_cols = feature_cols
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self):
        """Обучение множества моделей"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("="*60)
        
        # Определение моделей с гиперпараметрами
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=5, 
                min_samples_split=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=5,
                random_state=42
            ),
            'SVR': SVR(kernel='rbf', C=10, gamma='scale')
        }
        
        # Обучение и оценка каждой модели
        results = []
        
        for name, model in models_config.items():
            print(f"\n{'='*50}")
            print(f"Обучение: {name}")
            print(f"{'='*50}")
            
            # Обучение
            model.fit(self.X_train, self.y_train)
            
            # Предсказания
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Метрики на обучающей выборке
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            train_r2 = r2_score(self.y_train, y_pred_train)
            train_mape = mean_absolute_percentage_error(self.y_train, y_pred_train) * 100
            
            # Метрики на тестовой выборке
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            test_r2 = r2_score(self.y_test, y_pred_test)
            test_mape = mean_absolute_percentage_error(self.y_test, y_pred_test) * 100
            
            # Кросс-валидация
            if len(self.X_train) >= 5:
                tscv = TimeSeriesSplit(n_splits=min(3, len(self.X_train) // 2))
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                          cv=tscv, scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
            else:
                cv_mae = None
            
            # Сохранение результатов
            self.models[name] = model
            self.predictions[name] = y_pred_test
            self.metrics[name] = {
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'train_mape': train_mape,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_mape': test_mape,
                'cv_mae': cv_mae
            }
            
            results.append({
                'Model': name,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
                'Test RMSE': test_rmse,
                'Test R²': test_r2,
                'Test MAPE (%)': test_mape,
                'CV MAE': cv_mae if cv_mae else np.nan
            })
            
            print(f"\nОбучающая выборка:")
            print(f"  MAE: {train_mae:.2f} млрд тенге")
            print(f"  RMSE: {train_rmse:.2f} млрд тенге")
            print(f"  R²: {train_r2:.4f}")
            print(f"  MAPE: {train_mape:.2f}%")
            
            print(f"\nТестовая выборка:")
            print(f"  MAE: {test_mae:.2f} млрд тенге")
            print(f"  RMSE: {test_rmse:.2f} млрд тенге")
            print(f"  R²: {test_r2:.4f}")
            print(f"  MAPE: {test_mape:.2f}%")
            
            if cv_mae:
                print(f"\nКросс-валидация MAE: {cv_mae:.2f} млрд тенге")
        
        # Создание сводной таблицы
        results_df = pd.DataFrame(results)
        # Выбираем лучшую модель по минимальному Test MAPE (%)
        results_df = results_df.sort_values('Test MAPE (%)')
        
        print("\n" + "="*60)
        print("СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Выбор лучшей модели
        best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\n✓ Лучшая модель: {best_model_name}")
        print(f"  Test MAPE: {results_df.iloc[0]['Test MAPE (%)']:.2f}%")
        print(f"  Test R²: {results_df.iloc[0]['Test R²']:.4f}")
        
        return results_df
    
    def analyze_feature_importance(self):
        """Анализ важности признаков"""
        print("\n" + "="*60)
        print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("="*60)
        
        # Используем модели с feature importance
        importance_models = ['Random Forest', 'Gradient Boosting', 'Extra Trees']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, model_name in enumerate(importance_models):
            if model_name in self.models:
                model = self.models[model_name]
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]  # Топ-15
                
                ax = axes[idx]
                ax.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7)
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([self.feature_cols[i] for i in indices], fontsize=9)
                ax.set_xlabel('Важность', fontsize=11)
                ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('../main/output/feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Визуализация важности признаков сохранена в 'feature_importance.png'")
        
        # Вывод топ признаков для лучшей модели
        if self.best_model_name in importance_models:
            model = self.models[self.best_model_name]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            print(f"\nТоп-10 признаков для {self.best_model_name}:")
            for i, idx in enumerate(indices, 1):
                print(f"{i:2d}. {self.feature_cols[idx]:30s} - {importances[idx]:.4f}")
    
    def forecast_future(self, periods=3):
        """Прогноз на будущие периоды"""
        print("\n" + "="*60)
        print(f"ПРОГНОЗ НА {periods} МЕСЯЦА ВПЕРЁД")
        print("="*60)
        
        last_row = self.df.iloc[-1:].copy()
        current_month = int(last_row['month_num'].values[0])
        current_year = int(last_row['year'].values[0])
        current_index = int(last_row['month_index'].values[0])

        future_predictions = []
        future_dates = []

        for _ in range(periods):
            # Сдвиг на 1 месяц вперёд от текущего состояния
            if current_month == 12:
                next_month = 1
                next_year = current_year + 1
            else:
                next_month = current_month + 1
                next_year = current_year
            next_month_index = current_index + 1

            # Создание DataFrame признаков для следующего периода
            next_data = last_row.copy()
            next_data['month_num'] = next_month
            next_data['year'] = next_year
            next_data['month_index'] = next_month_index
            next_data['quarter'] = (next_month - 1) // 3 + 1
            next_data['is_year_end'] = int(next_month == 12)
            next_data['is_year_start'] = int(next_month == 1)
            next_data['is_quarter_end'] = int(next_month in [3, 6, 9, 12])
            next_data['month_sin'] = np.sin(2 * np.pi * next_month / 12)
            next_data['month_cos'] = np.cos(2 * np.pi * next_month / 12)

            # Прогноз
            next_features = next_data[self.feature_cols].values
            next_features_scaled = self.scaler.transform(next_features)
            prediction = self.best_model.predict(next_features_scaled)[0]

            future_predictions.append(prediction)
            future_dates.append(pd.Timestamp(year=int(next_year), month=int(next_month), day=1))

            # Обновление базовых значений для следующей итерации
            last_row = next_data.copy()
            last_row['amount_billion_tenge'] = prediction
            current_month = next_month
            current_year = next_year
            current_index = next_month_index
        
        # Вывод прогнозов
        month_names_en = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                         'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
        
        print(f"\nМодель: {self.best_model_name}")
        print("-" * 60)
        for date, pred in zip(future_dates, future_predictions):
            month_name = month_names_en[date.month - 1]
            print(f"{month_name} {date.year}: {pred:.2f} млрд тенге")
        
        return future_dates, future_predictions
    
    def visualize_results(self, future_dates=None, future_predictions=None):
        """Комплексная визуализация результатов"""
        print("\n" + "="*60)
        print("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("="*60)
        
        fig = plt.figure(figsize=(20, 12))
        
        # График 1: Сравнение предсказаний всех моделей
        ax1 = plt.subplot(2, 3, 1)
        test_dates = self.df['date'].iloc[-len(self.y_test):].values
        ax1.plot(test_dates, self.y_test, 'o-', label='Фактические', 
                linewidth=2.5, markersize=10, color='black', zorder=10)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.predictions)))
        for idx, (name, pred) in enumerate(self.predictions.items()):
            ax1.plot(test_dates, pred, 'o--', label=name, 
                    linewidth=1.5, markersize=6, alpha=0.7, color=colors[idx])
        
        ax1.set_title('Сравнение моделей на тестовой выборке', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Дата', fontsize=11)
        ax1.set_ylabel('Объём (млрд тенге)', fontsize=11)
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # График 2: Ошибки моделей
        ax2 = plt.subplot(2, 3, 2)
        model_names = list(self.metrics.keys())
        mae_values = [self.metrics[m]['test_mae'] for m in model_names]
        colors_bar = ['green' if m == self.best_model_name else 'steelblue' for m in model_names]
        
        ax2.barh(model_names, mae_values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('MAE (млрд тенге)', fontsize=11)
        ax2.set_title('Средняя абсолютная ошибка моделей', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # График 3: R² Score
        ax3 = plt.subplot(2, 3, 3)
        r2_values = [self.metrics[m]['test_r2'] for m in model_names]
        colors_bar = ['green' if m == self.best_model_name else 'coral' for m in model_names]
        
        ax3.barh(model_names, r2_values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('R² Score', fontsize=11)
        ax3.set_title('Коэффициент детерминации', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Порог 0.5')
        ax3.legend()
        
        # График 4: Остатки (residuals) лучшей модели
        ax4 = plt.subplot(2, 3, 4)
        residuals = self.y_test - self.predictions[self.best_model_name]
        ax4.scatter(self.predictions[self.best_model_name], residuals, 
                   alpha=0.6, s=100, color='purple', edgecolors='black')
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Предсказанные значения', fontsize=11)
        ax4.set_ylabel('Остатки', fontsize=11)
        ax4.set_title(f'Анализ остатков ({self.best_model_name})', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # График 5: Фактические vs Предсказанные
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(self.y_test, self.predictions[self.best_model_name], 
                   alpha=0.6, s=150, color='teal', edgecolors='black')
        min_val = min(self.y_test.min(), self.predictions[self.best_model_name].min())
        max_val = max(self.y_test.max(), self.predictions[self.best_model_name].max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальная линия')
        ax5.set_xlabel('Фактические значения', fontsize=11)
        ax5.set_ylabel('Предсказанные значения', fontsize=11)
        ax5.set_title(f'Accuracy Plot ({self.best_model_name})', fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # График 6: Прогноз на будущее
        ax6 = plt.subplot(2, 3, 6)
        all_dates = self.df['date'].values
        all_amounts = self.df['amount_billion_tenge'].values
        
        ax6.plot(all_dates, all_amounts, 'o-', label='Исторические данные', 
                linewidth=2.5, markersize=8, color='navy')
        
        if future_dates and future_predictions:
            ax6.plot(future_dates, future_predictions, 's--', label='Прогноз', 
                    color='red', linewidth=2.5, markersize=12)
            ax6.axvline(x=all_dates[-1], color='gray', linestyle='--', 
                       alpha=0.7, linewidth=2, label='Граница прогноза')
        
        ax6.set_title(f'Прогноз на будущее ({self.best_model_name})', fontsize=13, fontweight='bold')
        ax6.set_xlabel('Дата', fontsize=11)
        ax6.set_ylabel('Объём (млрд тенге)', fontsize=11)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../main/output/ml_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Комплексная визуализация сохранена в 'ml_comprehensive_analysis.png'")
        
        return fig


# Главная функция для запуска анализа
def run_ml_analysis(csv_file='bank_transfers_clean.csv'):
    """
    Запуск полного ML-анализа
    
    Parameters:
    -----------
    csv_file : str
        Путь к файлу с очищенными данными
    """
    print("\n" + "="*60)
    print("ЗАПУСК ПРОДВИНУТОГО ML-АНАЛИЗА")
    print("="*60)
    
    # Загрузка данных
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n✓ Загружено {len(df)} записей")
    
    # Создание анализатора
    analyzer = BankTransferMLAnalyzer(df)
    
    # Feature Engineering
    analyzer.engineer_features()
    
    # Подготовка данных
    analyzer.prepare_data(target='amount_billion_tenge', test_size=0.2)
    
    # Обучение моделей
    results_df = analyzer.train_models()
    
    # Анализ важности признаков
    analyzer.analyze_feature_importance()
    
    # Прогноз на будущее (окт 2025 - мар 2026 = 6 месяцев)
    future_dates, future_predictions = analyzer.forecast_future(periods=6)

    # Экспорт прогноза для фронтенда
    try:
        export_items = []
        month_names_ru = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                          'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
        for dt, val in zip(future_dates, future_predictions):
            export_items.append({
                'period': f"{month_names_ru[int(dt.month) - 1].lower()} {int(dt.year)}",
                'volumeBillionTenge': float(val),
                'isPrediction': True,
                'year': int(dt.year),
                'month': int(dt.month),
            })

        public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'front', 'public'))
        os.makedirs(public_dir, exist_ok=True)
        export_path = os.path.join(public_dir, 'forecast.json')
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model': analyzer.best_model_name,
                'items': export_items
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Прогноз экспортирован для фронтенда в '{export_path}'")
    except Exception as e:
        print(f"\n⚠️ Не удалось экспортировать прогноз для фронтенда: {e}")
    
    # Визуализация
    analyzer.visualize_results(future_dates, future_predictions)
    
    print("\n" + "="*60)
    print("ML-АНАЛИЗ ЗАВЕРШЁН УСПЕШНО!")
    print("="*60)
    
    return analyzer, results_df


if __name__ == "__main__":
    # Настройка matplotlib для русских символов
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Запуск анализа
    analyzer, results = run_ml_analysis('bank_transfers_clean.csv')