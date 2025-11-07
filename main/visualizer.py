"""
Модуль для создания визуализаций данных о межбанковских переводах
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TransferVisualizer:
    """Класс для визуализации данных межбанковских переводов"""
    
    def __init__(self, df):
        """
        Инициализация визуализатора
        
        Parameters:
        -----------
        df : pd.DataFrame
            Обработанный DataFrame с данными
        """
        self.df = df.copy()
        self.month_map = {
            'январь': 1, 'февраль': 2, 'март': 3, 'апрель': 4,
            'май': 5, 'июнь': 6, 'июль': 7, 'август': 8,
            'сентябрь': 9, 'октябрь': 10, 'ноябрь': 11, 'декабрь': 12
        }
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'seaborn-darkgrid')
        sns.set_palette("husl")
    
    def create_basic_visualizations(self):
        """Создание базовых визуализаций (6 графиков)"""
        print("\n   Создание базовых визуализаций...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # График 1: Динамика объёма переводов
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(self.df['date'], self.df['amount_billion_tenge'], 
                marker='o', linewidth=2.5, markersize=9, color='#2E86AB', 
                markerfacecolor='white', markeredgewidth=2)
        plt.title('Динамика объёма межбанковских переводов', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Дата', fontsize=12, fontweight='bold')
        plt.ylabel('Объём (млрд тенге)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Добавление тренда
        z = np.polyfit(range(len(self.df)), self.df['amount_billion_tenge'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['date'], p(range(len(self.df))), 
                "--", color='red', alpha=0.7, linewidth=2, label='Тренд')
        plt.legend(fontsize=10)
        
        # График 2: Динамика количества транзакций
        ax2 = plt.subplot(2, 3, 2)
        plt.plot(self.df['date'], self.df['transactions_thousand'], 
                marker='s', color='coral', linewidth=2.5, markersize=9,
                markerfacecolor='white', markeredgewidth=2)
        plt.title('Динамика количества транзакций', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Дата', fontsize=12, fontweight='bold')
        plt.ylabel('Количество (тыс. транзакций)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # График 3: Средний размер транзакции
        ax3 = plt.subplot(2, 3, 3)
        plt.plot(self.df['date'], self.df['avg_transaction_size'], 
                marker='^', color='green', linewidth=2.5, markersize=9,
                markerfacecolor='white', markeredgewidth=2)
        plt.title('Средний размер одной транзакции', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Дата', fontsize=12, fontweight='bold')
        plt.ylabel('Размер (тенге)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # График 4: Распределение объёмов
        ax4 = plt.subplot(2, 3, 4)
        n, bins, patches = plt.hist(self.df['amount_billion_tenge'], bins=8, 
                                    color='skyblue', edgecolor='black', alpha=0.7)
        # Раскраска столбцов по высоте
        cm = plt.cm.viridis
        for i, patch in enumerate(patches):
            patch.set_facecolor(cm(i / len(patches)))
        plt.title('Распределение объёмов переводов', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Объём (млрд тенге)', fontsize=12, fontweight='bold')
        plt.ylabel('Частота', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # График 5: Корреляция объём vs количество
        ax5 = plt.subplot(2, 3, 5)
        scatter = plt.scatter(self.df['transactions_thousand'], 
                            self.df['amount_billion_tenge'], 
                            s=150, alpha=0.6, c=self.df['month_num'], 
                            cmap='viridis', edgecolors='black', linewidth=1.5)
        cbar = plt.colorbar(scatter, label='Месяц')
        cbar.set_label('Месяц', fontsize=11, fontweight='bold')
        
        # Линия регрессии
        z = np.polyfit(self.df['transactions_thousand'], self.df['amount_billion_tenge'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['transactions_thousand'], 
                p(self.df['transactions_thousand']), 
                "r--", alpha=0.8, linewidth=2, label='Линия регрессии')
        
        plt.title('Связь между количеством и объёмом', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Количество транзакций (тыс.)', fontsize=12, fontweight='bold')
        plt.ylabel('Объём (млрд тенге)', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # График 6: Средний объём по месяцам
        ax6 = plt.subplot(2, 3, 6)
        monthly_avg = self.df.groupby('month')['amount_billion_tenge'].mean().reindex(list(self.month_map.keys()))
        bars = plt.bar(range(len(monthly_avg)), monthly_avg.values, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Раскраска столбцов
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(monthly_avg)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xticks(range(len(monthly_avg)), monthly_avg.index, rotation=45, ha='right', fontsize=10)
        plt.title('Средний объём по месяцам', fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('Объём (млрд тенге)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('../main/output/bank_transfers_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✓ Базовые визуализации сохранены в 'bank_transfers_analysis.png'")
        
        return fig
    
    def create_advanced_visualizations(self):
        """Создание продвинутых визуализаций"""
        print("\n   Создание продвинутых визуализаций...")
        
        fig = plt.figure(figsize=(20, 14))
        
        # График 1: Тепловая карта по месяцам и годам
        ax1 = plt.subplot(2, 3, 1)
        pivot_data = self.df.pivot_table(
            values='amount_billion_tenge', 
            index='month_num', 
            columns='year', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Объём (млрд тенге)'}, linewidths=0.5)
        plt.title('Тепловая карта объёмов по месяцам и годам', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Год', fontsize=12, fontweight='bold')
        plt.ylabel('Месяц', fontsize=12, fontweight='bold')
        
        # График 2: Box plot по кварталам
        ax2 = plt.subplot(2, 3, 2)
        bp = plt.boxplot([self.df[self.df['quarter'] == q]['amount_billion_tenge'] 
                          for q in range(1, 5)],
                         labels=['Q1', 'Q2', 'Q3', 'Q4'],
                         patch_artist=True,
                         notch=True,
                         showmeans=True)
        
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('Распределение объёмов по кварталам', fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('Объём (млрд тенге)', fontsize=12, fontweight='bold')
        plt.xlabel('Квартал', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # График 3: Кумулятивный рост
        ax3 = plt.subplot(2, 3, 3)
        cumulative_amount = self.df['amount_billion_tenge'].cumsum()
        plt.fill_between(self.df['date'], cumulative_amount, alpha=0.3, color='blue')
        plt.plot(self.df['date'], cumulative_amount, 
                marker='o', linewidth=2.5, color='darkblue', markersize=8)
        plt.title('Кумулятивный объём переводов', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Дата', fontsize=12, fontweight='bold')
        plt.ylabel('Кумулятивный объём (млрд тенге)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # График 4: Темп роста (MoM)
        ax4 = plt.subplot(2, 3, 4)
        growth_rate = self.df['amount_billion_tenge'].pct_change() * 100
        colors_growth = ['green' if x > 0 else 'red' for x in growth_rate]
        plt.bar(self.df['date'], growth_rate, color=colors_growth, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.title('Темп роста объёма (месяц к месяцу)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Дата', fontsize=12, fontweight='bold')
        plt.ylabel('Изменение (%)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # График 5: Violin plot
        ax5 = plt.subplot(2, 3, 5)
        data_by_quarter = [self.df[self.df['quarter'] == q]['amount_billion_tenge'].values 
                          for q in range(1, 5)]
        parts = plt.violinplot(data_by_quarter, positions=range(1, 5), 
                              showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        plt.title('Violin Plot по кварталам', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Квартал', fontsize=12, fontweight='bold')
        plt.ylabel('Объём (млрд тенге)', fontsize=12, fontweight='bold')
        plt.xticks(range(1, 5), ['Q1', 'Q2', 'Q3', 'Q4'])
        plt.grid(True, alpha=0.3, axis='y')
        
        # График 6: Корреляционная матрица
        ax6 = plt.subplot(2, 3, 6)
        corr_cols = ['amount_billion_tenge', 'transactions_thousand', 
                    'avg_transaction_size', 'month_num', 'quarter']
        corr_matrix = self.df[corr_cols].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Корреляционная матрица признаков', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig('../main/output/bank_transfers_advanced.png', dpi=300, bbox_inches='tight')
        print("   ✓ Продвинутые визуализации сохранены в 'bank_transfers_advanced.png'")
        
        return fig
    
    def create_time_series_analysis(self):
        """Анализ временных рядов"""
        print("\n   Создание анализа временных рядов...")
        
        fig = plt.figure(figsize=(20, 10))
        
        # График 1: Декомпозиция тренда
        ax1 = plt.subplot(2, 2, 1)
        
        # Скользящее среднее
        ma_3 = self.df['amount_billion_tenge'].rolling(window=3, center=True).mean()
        ma_5 = self.df['amount_billion_tenge'].rolling(window=5, center=True).mean()
        
        plt.plot(self.df['date'], self.df['amount_billion_tenge'], 
                'o-', label='Факт', linewidth=2, markersize=8, alpha=0.7)
        plt.plot(self.df['date'], ma_3, '--', label='MA(3)', linewidth=2.5)
        plt.plot(self.df['date'], ma_5, '--', label='MA(5)', linewidth=2.5)
        plt.title('Скользящие средние', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Дата', fontsize=12)
        plt.ylabel('Объём (млрд тенге)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # График 2: Сезонность
        ax2 = plt.subplot(2, 2, 2)
        monthly_pattern = self.df.groupby('month_num')['amount_billion_tenge'].agg(['mean', 'std'])
        x = monthly_pattern.index
        plt.errorbar(x, monthly_pattern['mean'], yerr=monthly_pattern['std'], 
                    fmt='o-', linewidth=2.5, markersize=10, capsize=5, capthick=2)
        plt.title('Сезонный паттерн с доверительным интервалом', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Месяц', fontsize=12)
        plt.ylabel('Средний объём (млрд тенге)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, 13))
        
        # График 3: Автокорреляция
        ax3 = plt.subplot(2, 2, 3)
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(self.df['amount_billion_tenge'])
        plt.title('Автокорреляционная функция', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Лаг', fontsize=12)
        plt.ylabel('Автокорреляция', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # График 4: QQ-plot
        ax4 = plt.subplot(2, 2, 4)
        stats.probplot(self.df['amount_billion_tenge'], dist="norm", plot=plt)
        plt.title('Q-Q Plot (проверка нормальности)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Теоретические квантили', fontsize=12)
        plt.ylabel('Выборочные квантили', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../main/output/time_series_analysis.png', dpi=300, bbox_inches='tight')
        print("   ✓ Анализ временных рядов сохранён в 'time_series_analysis.png'")
        
        return fig
    
    def create_all_visualizations(self):
        """Создание всех визуализаций"""
        print("\nСОЗДАНИЕ ВИЗУАЛИЗАЦИЙ:")
        print("-" * 60)
        
        # Базовые визуализации
        fig1 = self.create_basic_visualizations()
        
        # Продвинутые визуализации
        fig2 = self.create_advanced_visualizations()
        
        # Анализ временных рядов
        if len(self.df) >= 10:
            fig3 = self.create_time_series_analysis()
        else:
            print("\n   ⚠️ Недостаточно данных для анализа временных рядов (нужно ≥10 точек)")
        
        print("\n" + "="*60)
        print("ВСЕ ВИЗУАЛИЗАЦИИ СОЗДАНЫ УСПЕШНО!")
        print("="*60)


if __name__ == "__main__":
    # Тестирование модуля
    # Загрузка данных
    df = pd.read_csv('bank_transfers_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Создание визуализаций
    visualizer = TransferVisualizer(df)
    visualizer.create_all_visualizations()
    
    plt.show()