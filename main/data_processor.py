"""
Модуль для обработки и очистки данных о межбанковских переводах
"""

import pandas as pd
import numpy as np
from datetime import datetime


class DataProcessor:
    """Класс для обработки данных межбанковских переводов"""
    
    def __init__(self, json_data):
        """
        Инициализация процессора данных
        
        Parameters:
        -----------
        json_data : list
            Список словарей с исходными данными
        """
        self.json_data = json_data
        self.df = None
        self.month_map = {
            'январь': 1, 'января': 1,
            'февраль': 2, 'февраля': 2,
            'март': 3, 'марта': 3,
            'апрель': 4, 'апреля': 4,
            'май': 5, 'мая': 5,
            'июнь': 6, 'июня': 6,
            'июль': 7, 'июля': 7,
            'август': 8, 'августа': 8,
            'сентябрь': 9, 'сентября': 9,
            'октябрь': 10, 'октября': 10,
            'ноябрь': 11, 'ноября': 11,
            'декабрь': 12, 'декабря': 12
        }
    
    def process(self):
        """
        Основная функция обработки данных
        
        Returns:
        --------
        pd.DataFrame
            Обработанный DataFrame
        """
        # Создание DataFrame
        df = pd.DataFrame(self.json_data)
        
        print("\n1. Загрузка исходных данных:")
        print(f"   Размер: {df.shape}")
        print(f"   Столбцы: {list(df.columns)}")
        
        # Переименование столбцов
        df = self._rename_columns(df)
        
        # Преобразование типов
        df = self._convert_types(df)
        
        # Извлечение даты
        df = self._extract_date_features(df)
        
        # Очистка данных
        df = self._clean_data(df)
        
        # Создание дополнительных признаков
        df = self._create_features(df)
        
        # Сортировка
        df = df.sort_values('date').reset_index(drop=True)
        
        self.df = df
        
        print("\n2. Обработанные данные:")
        print(df[['date', 'amount_billion_tenge', 'transactions_thousand', 'avg_transaction_size']].head(10))
        
        return df
    
    def _rename_columns(self, df):
        """Переименование столбцов"""
        df = df.rename(columns={
            'name_ru': 'period',
            'transaction': 'transactions_thousand',
            'quantity': 'amount_million_tenge'
        })
        return df
    
    def _convert_types(self, df):
        """Преобразование типов данных"""
        print("\n   Преобразование типов данных...")
        
        # Числовые преобразования
        df['transactions_thousand'] = pd.to_numeric(df['transactions_thousand'], errors='coerce')
        df['amount_million_tenge'] = pd.to_numeric(df['amount_million_tenge'], errors='coerce')
        
        # Конвертация в миллиарды
        df['amount_billion_tenge'] = df['amount_million_tenge'] / 1000
        
        print(f"   ✓ Преобразовано {len(df.columns)} столбцов")
        
        return df
    
    def _extract_date_features(self, df):
        """Извлечение признаков даты"""
        print("\n   Извлечение временных признаков...")
        
        # Извлечение месяца и года
        df['month_raw'] = df['period'].str.extract(r'([а-яёА-ЯЁ]+)')[0]
        df['month'] = df['month_raw'].str.lower().str.strip()
        df['year'] = pd.to_numeric(df['period'].str.extract(r'(\d{4})')[0], errors='coerce')
        
        # Маппинг месяцев
        df['month_num'] = df['month'].map(self.month_map)
        
        # Проверка проблемных строк
        if df['month_num'].isna().any() or df['year'].isna().any():
            print("\n   ⚠️ ВНИМАНИЕ! Обнаружены проблемы с данными:")
            problem_rows = df[(df['month_num'].isna()) | (df['year'].isna())]
            for idx, row in problem_rows.iterrows():
                print(f"      Строка {idx}: period='{row['period']}', month='{row['month']}', year={row['year']}")
            
            print("\n   Удаляем проблемные строки...")
            df = df.dropna(subset=['month_num', 'year'])
            print(f"   ✓ Осталось строк: {len(df)}")
        
        # Преобразование типов
        df['year'] = df['year'].astype(int)
        df['month_num'] = df['month_num'].astype(int)
        
        # Создание даты
        try:
            df['date'] = pd.to_datetime(
                df['year'].astype(str) + '-' + 
                df['month_num'].astype(str).str.zfill(2) + '-01'
            )
            print(f"   ✓ Создан столбец 'date'")
        except Exception as e:
            print(f"\n   ✗ Ошибка при создании дат: {e}")
            raise
        
        return df
    
    def _clean_data(self, df):
        """Очистка данных от дубликатов и пропусков"""
        print("\n   Очистка данных...")
        
        initial_size = len(df)
        
        # Удаление дубликатов
        df = df.drop_duplicates()
        duplicates_removed = initial_size - len(df)
        
        # Удаление пропусков
        df = df.dropna(subset=['amount_billion_tenge', 'transactions_thousand'])
        missing_removed = initial_size - duplicates_removed - len(df)
        
        if duplicates_removed > 0 or missing_removed > 0:
            print(f"   ✓ Удалено дубликатов: {duplicates_removed}")
            print(f"   ✓ Удалено пропусков: {missing_removed}")
        else:
            print(f"   ✓ Данные чистые (дубликатов и пропусков нет)")
        
        return df
    
    def _create_features(self, df):
        """Создание дополнительных признаков"""
        print("\n   Создание дополнительных признаков...")
        
        # Индекс месяца
        df['month_index'] = range(len(df))
        
        # Средний размер транзакции
        df['avg_transaction_size'] = (
            (df['amount_billion_tenge'] * 1_000_000_000) / 
            (df['transactions_thousand'] * 1000)
        )
        
        # Квартал
        df['quarter'] = df['month_num'].apply(lambda x: (x-1)//3 + 1)
        
        # Признаки конца/начала периода
        df['is_year_end'] = (df['month_num'] == 12).astype(int)
        df['is_year_start'] = (df['month_num'] == 1).astype(int)
        df['is_quarter_end'] = df['month_num'].isin([3, 6, 9, 12]).astype(int)
        
        print(f"   ✓ Создано признаков: {df.shape[1]}")
        
        return df
    
    def print_statistics(self):
        """Вывод описательной статистики"""
        if self.df is None:
            print("Данные не обработаны. Запустите process() сначала.")
            return
        
        df = self.df
        
        print("\nОСНОВНЫЕ ПОКАЗАТЕЛИ:")
        print("-" * 60)
        
        stats_df = df[['amount_billion_tenge', 'transactions_thousand', 'avg_transaction_size']].describe()
        print(stats_df)
        
        print("\n\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
        print("-" * 60)
        print(f"Период анализа: {df['date'].min().strftime('%B %Y')} - {df['date'].max().strftime('%B %Y')}")
        print(f"Всего месяцев: {len(df)}")
        print(f"\nОбщий объём переводов: {df['amount_billion_tenge'].sum():.2f} млрд тенге")
        print(f"Общее количество транзакций: {df['transactions_thousand'].sum():.2f} тысяч")
        print(f"Средний размер транзакции: {df['avg_transaction_size'].mean():.2f} тенге")
        
        # Медианы
        print(f"\nМедианный объём: {df['amount_billion_tenge'].median():.2f} млрд тенге")
        print(f"Медианное количество транзакций: {df['transactions_thousand'].median():.2f} тысяч")
        
        # Разброс
        print(f"\nДиапазон объёмов: {df['amount_billion_tenge'].min():.2f} - {df['amount_billion_tenge'].max():.2f} млрд")
        print(f"Стандартное отклонение объёма: {df['amount_billion_tenge'].std():.2f} млрд")
        
        # Корреляционная матрица
        print("\n\nКОРРЕЛЯЦИОННАЯ МАТРИЦА:")
        print("-" * 60)
        corr_cols = ['amount_billion_tenge', 'transactions_thousand', 'avg_transaction_size']
        corr_matrix = df[corr_cols].corr()
        print(corr_matrix)
        
        # Ключевые инсайты
        print("\n\nКЛЮЧЕВЫЕ НАБЛЮДЕНИЯ:")
        print("-" * 60)
        
        # Тренд
        first_value = df['amount_billion_tenge'].iloc[0]
        last_value = df['amount_billion_tenge'].iloc[-1]
        change_pct = ((last_value / first_value) - 1) * 100
        
        print(f"1. Изменение объёма за период: {change_pct:+.1f}%")
        
        # Волатильность
        cv = (df['amount_billion_tenge'].std() / df['amount_billion_tenge'].mean()) * 100
        print(f"2. Коэффициент вариации: {cv:.1f}% ({'высокая' if cv > 15 else 'умеренная' if cv > 10 else 'низкая'} волатильность)")
        
        # Сезонность
        monthly_avg = df.groupby('month_num')['amount_billion_tenge'].mean()
        seasonal_range = monthly_avg.max() - monthly_avg.min()
        print(f"3. Сезонный разброс: {seasonal_range:.2f} млрд тенге")
        
        # Корреляция
        corr = df['amount_billion_tenge'].corr(df['transactions_thousand'])
        print(f"4. Корреляция объём-количество: {corr:.4f} ({'сильная' if abs(corr) > 0.7 else 'умеренная' if abs(corr) > 0.4 else 'слабая'})")
        
    def get_summary(self):
        """Получение краткой сводки по данным"""
        if self.df is None:
            return None
        
        df = self.df
        
        summary = {
            'total_records': len(df),
            'date_range': (df['date'].min(), df['date'].max()),
            'total_amount': df['amount_billion_tenge'].sum(),
            'total_transactions': df['transactions_thousand'].sum(),
            'avg_transaction_size': df['avg_transaction_size'].mean(),
            'peak_month': df.loc[df['amount_billion_tenge'].idxmax(), 'period'],
            'lowest_month': df.loc[df['amount_billion_tenge'].idxmin(), 'period'],
            'volatility': (df['amount_billion_tenge'].std() / df['amount_billion_tenge'].mean()) * 100,
            'trend': 'рост' if df['amount_billion_tenge'].iloc[-1] > df['amount_billion_tenge'].iloc[0] else 'снижение'
        }
        
        return summary
    
    def export_summary_report(self, filename='summary_report.txt'):
        """Экспорт текстового отчёта"""
        if self.df is None:
            print("Данные не обработаны.")
            return
        
        summary = self.get_summary()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("СВОДНЫЙ ОТЧЁТ ПО МЕЖБАНКОВСКИМ ПЕРЕВОДАМ КАЗАХСТАНА\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Дата формирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("ОСНОВНЫЕ ПОКАЗАТЕЛИ:\n")
            f.write("-"*60 + "\n")
            f.write(f"Период анализа: {summary['date_range'][0].strftime('%B %Y')} - {summary['date_range'][1].strftime('%B %Y')}\n")
            f.write(f"Всего записей: {summary['total_records']}\n")
            f.write(f"Общий объём переводов: {summary['total_amount']:.2f} млрд тенге\n")
            f.write(f"Общее количество транзакций: {summary['total_transactions']:.2f} тысяч\n")
            f.write(f"Средний размер транзакции: {summary['avg_transaction_size']:.2f} тенге\n\n")
            
            f.write("ЭКСТРЕМУМЫ:\n")
            f.write("-"*60 + "\n")
            f.write(f"Пиковый месяц: {summary['peak_month']}\n")
            f.write(f"Минимальный месяц: {summary['lowest_month']}\n\n")
            
            f.write("ДИНАМИКА:\n")
            f.write("-"*60 + "\n")
            f.write(f"Общий тренд: {summary['trend']}\n")
            f.write(f"Волатильность: {summary['volatility']:.1f}%\n")
        
        print(f"\n✓ Сводный отчёт сохранён в '{filename}'")


if __name__ == "__main__":
    json_data = [
        {"id": "4", "transaction": "2682.9119999999948", "name_ru": "март 2025 года", 
         "quantity": "109585.26789273391", "name_kz": "2025 жылғы наурыз"},
    ]
    
    processor = DataProcessor(json_data)
    df = processor.process()
    processor.print_statistics()
    processor.export_summary_report()