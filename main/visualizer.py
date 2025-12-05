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

    def create_all_visualizations(self):
        """Создание всех визуализаций"""
        print("\nСОЗДАНИЕ ВИЗУАЛИЗАЦИЙ:")
        print("-" * 60)
        
        # Только базовые визуализации
        fig1 = self.create_basic_visualizations()
        
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