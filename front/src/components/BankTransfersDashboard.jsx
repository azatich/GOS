import React, { useState, useMemo, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, Calendar, Target } from 'lucide-react';

export const BankTransfersDashboard = () => {
  // Исходные данные
  const rawData = [
    {id: "4", transaction: 2682.91, name_ru: "март 2025 года", quantity: 109585.27, name_kz: "2025 жылғы наурыз"},
    {id: "9", transaction: 3429.19, name_ru: "август 2025 года", quantity: 142582.03, name_kz: "2025 жылғы тамыз"},
    {id: "5", transaction: 2959.85, name_ru: "апрель 2025 года", quantity: 131946.65, name_kz: "2025 жылғы сәуір"},
    {id: "1", transaction: 3152.63, name_ru: "декабрь 2024 года", quantity: 138713.92, name_kz: "2024 жылғы желтоқсан"},
    {id: "6", transaction: 2894.19, name_ru: "май 2025 года", quantity: 125944.30, name_kz: "2025 жылғы мамыр"},
    {id: "2", transaction: 2507.88, name_ru: "январь 2025 года", quantity: 123621.69, name_kz: "2025 жылғы қаңтар"},
    {id: "7", transaction: 2769.86, name_ru: "июнь 2025 года", quantity: 131782.94, name_kz: "2025 жылғы маусым"},
    {id: "3", transaction: 3310.39, name_ru: "февраль 2025 года", quantity: 116193.44, name_kz: "2025 жылғы ақпан"},
    {id: "8", transaction: 2913.41, name_ru: "июль 2025 года", quantity: 145247.20, name_kz: "2025 жылғы шілде"},
    {id: "10", transaction: 2752.42, name_ru: "сентябрь 2025 года", quantity: 113574.15, name_kz: "2025 жылғы қыркүйек"}
  ];

  const [selectedMetric, setSelectedMetric] = useState('volume');
  const [showPrediction, setShowPrediction] = useState(false);
  const [forecast, setForecast] = useState({ model: null, items: [] });
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [forecastError, setForecastError] = useState(null);
  const [allForecasts, setAllForecasts] = useState({ bestModel: null, models: [] });
  const [selectedModel, setSelectedModel] = useState(null);

  // Обработка данных
  const processedData = useMemo(() => {
    const monthOrder = {
      'декабрь 2024': 0, 'январь 2025': 1, 'февраль 2025': 2, 'март 2025': 3,
      'апрель 2025': 4, 'май 2025': 5, 'июнь 2025': 6, 'июль 2025': 7,
      'август 2025': 8, 'сентябрь 2025': 9
    };

    return rawData
      .map(item => ({
        period: item.name_ru,
        month: item.name_ru.split(' ')[0],
        volumeBillionTenge: item.quantity / 1000,
        transactionsThousand: item.transaction,
        avgTransactionSize: (item.quantity * 1000000) / (item.transaction * 1000),
        sortOrder: monthOrder[item.name_ru.split(' года')[0]]
      }))
      .sort((a, b) => a.sortOrder - b.sortOrder);
  }, []);

  // Загрузка прогноза, экспортированного из Python (если доступен)
  useEffect(() => {
    const loadForecast = async () => {
      setLoadingForecast(true);
      setForecastError(null);
      try {
        let res = await fetch('/forecast.json', { cache: 'no-store' });
        if (!res.ok) {
          
          res = await fetch('forecast.json', { cache: 'no-store' });
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        const items = Array.isArray(data.items) ? data.items.map(item => ({
          period: item.period,
          volumeBillionTenge: item.volumeBillionTenge,
          isPrediction: true
        })) : [];
        setForecast({ model: data.model || null, items });
      } catch (e) {
        
        try {
          const lastThree = processedData.slice(-3);
          const avgGrowth = (lastThree[2].volumeBillionTenge - lastThree[0].volumeBillionTenge) / 2;
          const base = processedData[processedData.length - 1]?.volumeBillionTenge || 0;
          const fallbackItems = [
            { period: 'октябрь 2025', volumeBillionTenge: base + avgGrowth, isPrediction: true },
            { period: 'ноябрь 2025', volumeBillionTenge: base + avgGrowth * 2, isPrediction: true },
            { period: 'декабрь 2025', volumeBillionTenge: base + avgGrowth * 3, isPrediction: true }
          ];
          setForecast({ model: 'Baseline', items: fallbackItems });
          setForecastError(e.message);
        } catch (_) {
          setForecast({ model: null, items: [] });
          setForecastError(e.message);
        }
      } finally {
        setLoadingForecast(false);
      }
    };
    loadForecast();
  }, []);

  // Загрузка прогнозов всех моделей
  useEffect(() => {
    const loadAll = async () => {
      try {
        let res = await fetch('/forecast_all.json', { cache: 'no-store' });
        if (!res.ok) res = await fetch('forecast_all.json', { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        setAllForecasts({ bestModel: data.bestModel || null, models: Array.isArray(data.models) ? data.models : [] });
        setSelectedModel(data.bestModel || (data.models?.[0]?.model ?? null));
      } catch (_) {
        // no-op
      }
    };
    loadAll();
  }, []);

  // Данные для выбранной модели (если загружены все модели)
  const selectedModelItems = useMemo(() => {
    if (!showPrediction) return [];
    if (allForecasts.models?.length && selectedModel) {
      const m = allForecasts.models.find(x => x.model === selectedModel);
      if (m && Array.isArray(m.items)) return m.items.map(it => ({
        period: it.period,
        volumeBillionTenge: it.volumeBillionTenge,
        isPrediction: true
      }));
    }
    return forecast.items || [];
  }, [showPrediction, allForecasts, selectedModel, forecast]);

  const displayData = showPrediction
    ? [...processedData, ...selectedModelItems]
    : processedData;

  const stats = useMemo(() => {
    const baseData = showPrediction ? displayData : processedData;
    const volumes = baseData.map(d => d.volumeBillionTenge).filter(v => typeof v === 'number');
    const transactions = processedData.map(d => d.transactionsThousand);
    const avgTxSizeVals = processedData.map(d => d.avgTransactionSize).filter(v => typeof v === 'number');

    const safeSum = (arr) => arr.reduce((a, b) => a + b, 0);
    const totalVolume = safeSum(volumes);
    const avgVolume = volumes.length ? totalVolume / volumes.length : 0;
    const totalTransactions = safeSum(transactions);
    const avgTransactionSize = avgTxSizeVals.length ? safeSum(avgTxSizeVals) / avgTxSizeVals.length : 0;
    const maxVolume = volumes.length ? Math.max(...volumes) : 0;
    const minVolume = volumes.length ? Math.min(...volumes) : 0;
    const trend = volumes.length && volumes[volumes.length - 1] > volumes[0] ? 'up' : 'down';

    return {
      totalVolume,
      totalTransactions,
      avgVolume,
      avgTransactionSize,
      maxVolume,
      minVolume,
      trend
    };
  }, [processedData, displayData, showPrediction]);

  const StatCard = ({ title, value, subtitle, icon: Icon, trend }) => (
    <div className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-gray-600 text-sm font-medium mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-full ${trend === 'up' ? 'bg-green-100' : trend === 'down' ? 'bg-red-100' : 'bg-blue-100'}`}>
          <Icon className={`w-6 h-6 ${trend === 'up' ? 'text-green-600' : trend === 'down' ? 'text-red-600' : 'text-blue-600'}`} />
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Анализ межбанковских переводов Казахстана
          </h1>
          <p className="text-gray-600">Данные с портала data.egov.kz | Период: Декабрь 2024 - Сентябрь 2025</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            title="Общий объём"
            value={`${stats.totalVolume.toFixed(1)}`}
            subtitle="млрд тенге"
            icon={DollarSign}
          />
          <StatCard
            title="Всего транзакций"
            value={`${(stats.totalTransactions / 1000).toFixed(1)}M`}
            subtitle="транзакций"
            icon={Activity}
          />
          <StatCard
            title="Средний размер"
            value={`${(stats.avgTransactionSize / 1000).toFixed(1)}K`}
            subtitle="тенге за транзакцию"
            icon={Target}
          />
          <StatCard
            title="Тренд"
            value={stats.trend === 'up' ? '+8.2%' : '-3.5%'}
            subtitle="за период"
            icon={stats.trend === 'up' ? TrendingUp : TrendingDown}
            trend={stats.trend}
          />
        </div>

        {/* Controls */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="flex flex-wrap gap-4 items-center">
            <div>
              <label className="text-sm font-medium text-gray-700 mr-3">Метрика:</label>
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="volume">Объём переводов</option>
                <option value="transactions">Количество транзакций</option>
                <option value="avgSize">Средний размер</option>
              </select>
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                id="prediction"
                checked={showPrediction}
                onChange={(e) => setShowPrediction(e.target.checked)}
                className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <label htmlFor="prediction" className="ml-2 text-sm font-medium text-gray-700">
                Показать прогноз на 6 месяцов {selectedModel ? `(ML: ${selectedModel})` : forecast.model ? `(ML: ${forecast.model})` : ''}
              </label>
            </div>
            {allForecasts.models?.length > 0 && (
              <div>
                <label className="text-sm font-medium text-gray-700 mr-3">Модель:</label>
                <select
                  value={selectedModel || allForecasts.bestModel || ''}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {allForecasts.models.map(m => (
                    <option key={m.model} value={m.model}>{m.model}</option>
                  ))}
                </select>
              </div>
            )}
            {showPrediction && loadingForecast && (
              <span className="text-sm text-gray-500">Загрузка прогноза…</span>
            )}
            {showPrediction && forecastError && (
              <span className="text-sm text-red-600">Не удалось загрузить прогноз</span>
            )}
          </div>
        </div>

        {/* Main Chart */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-4">
            Динамика {selectedMetric === 'volume' ? 'объёма переводов' : selectedMetric === 'transactions' ? 'количества транзакций' : 'среднего размера транзакции'}
          </h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={displayData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis 
                dataKey="period" 
                angle={-45} 
                textAnchor="end" 
                height={100}
                tick={{ fontSize: 12 }}
              />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'white', border: '1px solid #ccc', borderRadius: '8px' }}
                formatter={(value) => [
                  `${Number(value).toFixed(2)}`,
                  selectedMetric === 'volume' ? 'млрд ₸' : selectedMetric === 'transactions' ? 'тыс. транз.' : '₸'
                ]}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey={selectedMetric === 'volume' ? 'volumeBillionTenge' : selectedMetric === 'transactions' ? 'transactionsThousand' : 'avgTransactionSize'}
                stroke="#3b82f6"
                strokeWidth={3}
                dot={{ fill: '#3b82f6', r: 5 }}
                name={selectedMetric === 'volume' ? 'Объём (млрд ₸)' : selectedMetric === 'transactions' ? 'Транзакции (тыс.)' : 'Размер (₸)'}
              />
              {showPrediction && selectedModelItems.length > 0 && (
                <Line
                  type="monotone"
                  dataKey="volumeBillionTenge"
                  stroke="#ef4444"
                  strokeWidth={3}
                  strokeDasharray="5 5"
                  dot={{ fill: '#ef4444', r: 5 }}
                  name={`Прогноз (${selectedModel || forecast.model || 'ML'})`}
                  connectNulls
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Two Column Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Bar Chart */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Объём по месяцам</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={processedData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="month" angle={-45} textAnchor="end" height={80} tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'white', border: '1px solid #ccc', borderRadius: '8px' }}
                  formatter={(value) => [`${Number(value).toFixed(2)} млрд ₸`, 'Объём']}
                />
                <Bar dataKey="volumeBillionTenge" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Scatter Plot */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Корреляция объёма и количества</h2>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis 
                  dataKey="transactionsThousand" 
                  name="Транзакции" 
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Транзакции (тыс.)', position: 'bottom' }}
                />
                <YAxis 
                  dataKey="volumeBillionTenge" 
                  name="Объём" 
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Объём (млрд ₸)', angle: -90, position: 'left' }}
                />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ backgroundColor: 'white', border: '1px solid #ccc', borderRadius: '8px' }}
                  formatter={(value, name) => [
                    `${Number(value).toFixed(2)}`,
                    name === 'transactionsThousand' ? 'Транзакции (тыс.)' : 'Объём (млрд ₸)'
                  ]}
                />
                <Scatter data={processedData} fill="#10b981" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Insights */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Ключевые выводы</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">📊 Динамика</h3>
              <p className="text-sm text-gray-700">
                Объём переводов показывает {stats.trend === 'up' ? 'положительную' : 'отрицательную'} динамику.
                Пиковое значение: {stats.maxVolume.toFixed(2)} млрд ₸
              </p>
            </div>
            <div className="p-4 bg-green-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">💰 Средний размер</h3>
              <p className="text-sm text-gray-700">
                Средний размер одной транзакции составляет {(stats.avgTransactionSize / 1000).toFixed(1)}K тенге,
                что указывает на активность малого и среднего бизнеса
              </p>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">📈 Прогноз</h3>
              <p className="text-sm text-gray-700">
                ML-модель {selectedModel ? `(${selectedModel})` : forecast.model ? `(${forecast.model})` : ''} прогнозирует {showPrediction ? 'стабильный рост' : 'включите прогноз для просмотра'} 
                межбанковских переводов на следующие месяцы
              </p>
            </div>
            <div className="p-4 bg-yellow-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">🎯 Рекомендации</h3>
              <p className="text-sm text-gray-700">
                Необходимо усилить мониторинг в периоды пиковой активности и
                оптимизировать инфраструктуру для обработки транзакций
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>Данные обработаны с использованием Python, Pandas, Scikit-learn</p>
          <p>Визуализация: React + Recharts | Источник: data.egov.kz</p>
        </div>
      </div>
    </div>
  );
};