# Week 1

1. 下载并处理netCDF4数据，输出一个包含特定经纬度的CSV格式数据，包括时间，温度，湿度，海平面压力以及我们所需预测的数据—降雨量。|
[把netCDF4文件转化为CSV文件.ipynb - Colaboratory (google.com)](https://colab.research.google.com/drive/1mOIKYZ4DrjF4IH5llI0KKn68sRWawW6p)
[替换列加入主csv文件.ipynb - Colaboratory (google.com)](https://colab.research.google.com/drive/1oySrXdK_yOxM0cGow9rHRk-zMchCL_oz)
2. 用这个简单的数据进行时间序列分析。包括平稳性检验等。
    - [Time Series data analysis.ipynb](https://colab.research.google.com/drive/1hu2gfUGSKEyGV9r5K6kG7mlPpKVz3aD7#scrollTo=ae4i6KHH211z)
3. 使用一个简单的机器学习模型进行训练以及预测。 
    - [RNN Method - Prediction of precipitation.ipynb](https://colab.research.google.com/drive/1JQJJaEqXEAFCjlkjuqsEx0Zn1ykAdDhi#scrollTo=umofRzBWlcsw)
4. 使用一个更好的模型来处理这个预测问题，在这里我选择了LSTM来处理。
    - [LSTM Method - Prediction of Precipitation.ipynb](https://colab.research.google.com/drive/1HE52cH7wIfjQl64MekuAGvopxx2SQtmu#scrollTo=ZlFnxgpCGl1j&uniqifier=1)
5. 在这里使用的数据还都是一个点位，即固定经纬度的一个点的时间序列气象数据。下一步需要将经纬度加进来，并且把不同经纬度的气象属性数据集依旧按照时间顺序排列。
6. 看论文，了解时间序列如何应用的Drought Prediction中。

| Paper | Summary |
| --- | --- |
| http://polyu.edu.hk | 1. 本文研究涉及两个时间序列，一个与 SPI 月度干旱指数数据相关，另一个与 SPI 参数的六个月数据相关。并且设计了不同的差分时间序列（包括季节性，非季节性，以及混合差分以形成新的时间序列）.2. This research utilizes ARIMA models and their general form, multiplicative seasonal ARIMA, to predict the drought index. Using assess metrics including: correlation coefficient (R),RMSE and MAE. 3. 一些小概念：Standardized Precipitation Index，Time Series Stationarity |
| https://research-repository.griffith.edu.au/bitstream/handle/10072/404189/Gyasi-Agyei483825-Published.pdf?sequence=2 | 1. 在这项研究中，建议将机器学习与标准化降水蒸散指数 (SPEI) 相结合，以分析 1980 年至 2019 年中国青藏高原代表性案例研究中的干旱情况。2. Four machine learning models of Random Forest (RF), the Extreme Gradient Boost (XGB), the Convolutional neural network (CNN) and the Long-term short memory (LSTM) 并比较这些模型在 SPEI 预测中的准确性和稳定性，并根据预测准确性选择最佳模型。3. 一些小概念：SPEI, Total Drought Duration (TDD), Drought Severity (DS), drought peak (DP) and Spatial Extent of Drought (SEoD)which were analyzed for specific drought events, Taylor diagrams4. 实验中首先将时间序列数据进行归一化，然后分别用RF, XGB, CNN 和LSTM的方法在七个气候输入组合的数据集分别训练模型。5. 评价指标: Nash–Sutcliffe model efficiency coefficient (NSE), MSE, MAE, mean bias error (MBE) and R. |
| https://www.scirp.org/pdf/jwarp_2021081015095566.pdf | 1. 本研究提出并评估了新开发的准确预测模型，该模型利用各种水文、气象和地质水文参数，以及使用具有各种预测提前期的人工神经网络 (ANN) 模型. 本研究的主要重点是构建并验证预测各种干旱情景的可行性和似是而非的准确性，包括次区域尺度的气象、水文和农业干旱情景。2. 干旱指标：PDSI、SPI、SMI 和 ENSO，文中均有简介3. 利用时间序列神经网络作为未来干旱预报情景的预测工具。特别是，具有外部输入的非线性自回归 (NARX) 模型被用作首选的神经网络工具。该模型在其过程中同时使用时间序列和回归分析。并用MSE来作为评估指标。 |
| https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7485508/ | 1. Propose an ensemble approach for monthly drought prediction and to define and examine wet/dry events.2. Ensemble models are divided into three categories based on several weighting techniques that are the EEDP, WEDP and CEDP model.3. 干旱是根据三个集合模型预测的，即等集合干旱预测（EEDP），加权集合干旱预测 (WEDP) 和条件集合干旱预测 (CEDP) 模型。4. 评估指标：MAE, MSE, RMSE, MAPE, NRMSE, AB test |
| https://link.springer.com/article/10.1007/s12652-022-03701-7 | 1. 本文介绍了一种新的混合智能模型，即用于短期气象干旱预报的卷积长短期记忆（CNN-LSTM）的开发和验证程序。CNN-LSTM将长短期记忆（LSTM）网络与卷积神经网络（CNN）作为特征提取器进行偶联。2. 在这项研究中，对3个月的SPEI（SPEI-3）和6个月的SPEI（SPEI-6）进行了建模和预测，时间为一个月的提前期。使用自相关函数（ACF）和部分自相关函数（PACF）确定最佳输入向量（最佳滞后）。以经典的ANN和遗传编程（GP）技术为基准，以及两种深度学习技术，即LSTM和CNN，作为基线，用于对相关的预测场景进行建模3. 评估指标：including absolute error measures(RMSE, MAE) , best-fit goodness (NSE: Nash–Sutcliffe coefficient, and WI: Willmott's index)4. 也是使用SPEI作为衡量干旱的 |- 这几篇文章中了解到包括干旱指标（PDSI、SPI、SMI 和 ENSO，SPEI等），一些可以应用在气象数据中的模型（包括传统时间序列方法，AR，ARMA，ARIMA等和与机器学习深度学习相结合的时间序列方法ANN，RNN，LSTM，CNN-LSTM，Transformer等），以及评估模型效果的指标（MAE, MSE, RMSE, MAPE, NRMSE, AB test）。- 在文章对气象数据的预处理中，我们可以了解到以下常见的时间序列分析。1. 类似于引导策略搜索[(23) Physics-Guided AI for Learning Spatiotemporal Dynamics - YouTube](https://www.youtube.com/watch?v=eGz2y3DB_Sw)2. 简单了解了Transformer的架构以及其与RNN的区别。

## 时间序列处理的一些统计模型

1. **AR（自回归模型）**：自回归模型假设当前的值可以由过去的几个值线性表示。例如，一个AR(1)模型表示当前值是上一个值的函数，一个AR(2)模型表示当前值是过去两个值的函数，以此类推。具体来说，一个AR(p)模型可以表示为：
    
    Yt = c + φ1*Yt-1 + φ2*Yt-2 + ... + φp*Yt-p + εt
    
    其中，Yt是当前值，Yt-1是前一期的值，以此类推，c是常数，φ1到φp是参数，εt是误差项。
    
2. **MA（移动平均模型）**：移动平均模型假设当前值是过去误差项的线性组合。例如，一个MA(1)模型表示当前值是上一个误差项的函数，一个MA(2)模型表示当前值是过去两个误差项的函数，以此类推。具体来说，一个MA(q)模型可以表示为：
    
    Yt = µ + εt + θ1*εt-1 + θ2*εt-2 + ... + θq*εt-q
    
    其中，Yt是当前值，εt是当前误差项，εt-1是前一期的误差项，以此类推，µ是常数，θ1到θq是参数。误差项通常是通过最小化预测值和实际观察值之间的差异（例如，通过最小化均方误差）来估计的。
    
3. **ARMA（自回归移动平均模型）**：自回归移动平均模型是AR和MA模型的结合。它假设当前值既依赖于过去的值，也依赖于过去的误差项。一个ARMA(p, q)模型可以表示为：
    
    Yt = c + φ1*Yt-1 + ... + φp*Yt-p + εt + θ1*εt-1 + ... + θq*εt-q
    
4. **ARIMA（自回归集成移动平均模型）**：自回归集成移动平均模型是ARMA模型的扩展，它增加了一个集成部分来处理非平稳时间序列。ARIMA模型首先对原始数据进行差分操作，以得到平稳时间序列，然后在平稳时间序列上应用ARMA模型。一个ARIMA(p, d, q)模型表示对原始数据进行d阶差分后，得到的平稳时间序列满足ARMA(p, q)模型。

这些模型都有其适用的场景和限制，选择哪种模型取决于你的数据和你的研究问题。例如，AR模型通常用于处理具有滞后关系的数据，MA模型则适合处理噪声或震荡较大的数据，ARMA和ARIMA模型则可以处理更复杂的情况。

AR、MA、ARMA和ARIMA模型的参数通常通过最大似然估计法进行估计，这需要使用专门的统计软件或包，如R的**`forecast`**包或Python的**`statsmodels`**库。

这些模型的预测性能需要通过适当的模型验证和预测误差度量来评估，例如均方误差（MSE）、均方根误差（RMSE）或平均绝对误差（MAE）等。

值得注意的是，这些模型都基于一些假设，例如，误差项通常假设为独立同分布（i.i.d.），并且通常假设为正态分布。在应用这些模型之前，应检查这些假设是否成立。如果假设不成立，可能需要对数据进行适当的转换，例如对数转换或Box-Cox转换等。

另外，虽然AR、MA、ARMA和ARIMA模型在许多情况下都非常有用，但在一些复杂的情况下，可能需要使用更复杂的模型，例如，季节ARIMA模型、状态空间模型、或基于机器学习的预测模型等。

## 时间序列预测问题的一些机器学习，深度学习模型

1. **长短期记忆网络（LSTM）**：LSTM是一种特殊类型的循环神经网络（RNN），它能够学习并记住长期序列的信息。LSTM解决了传统RNN在处理长序列时出现的梯度消失问题。在时间序列预测中，LSTM能够记住过去的信息并利用这些信息对未来进行预测。
2. **门控循环单元（GRU）**：GRU是另一种特殊类型的RNN，它类似于LSTM但是结构更简单。GRU也能处理长序列，但是它将LSTM的遗忘门和输入门合并成一个“更新门”。虽然GRU的参数少于LSTM，但在某些任务上，GRU的性能与LSTM相当。
3. **序列到序列模型（Seq2Seq）**：Seq2Seq模型是一种使用RNN（如LSTM或GRU）进行编码和解码的模型，主要用于机器翻译、问答系统等任务。在时间序列预测中，Seq2Seq模型可以将一段时间序列数据（例如过去几天的销售数据）编码为固定长度的向量，然后将这个向量解码为未来的预测（例如未来几天的销售量）。
4. **一维卷积神经网络（1D-CNN）**：1D-CNN是卷积神经网络的一种变体，它在一维空间（即时间轴）上进行卷积操作。1D-CNN可以学习时间序列数据的局部模式，并能够处理长度可变的序列。在时间序列预测任务中，1D-CNN可以捕捉到一段时间窗口内的模式，并用这些模式进行未来的预测。
5. **DeepAR**
    
    DeepAR 是 Amazon Web Services (AWS) 提供的一种预测模型，它使用循环神经网络 (RNN) 来处理大规模、复杂的时间序列数据。DeepAR 能够处理各种时间序列数据，并且可以利用已有的历史数据来进行预测。这种模型对于解决复杂的时间序列问题，如股票价格预测、销售量预测、电力需求预测等非常有用。
    
    在 DeepAR 模型中，使用循环神经网络 (RNN) 学习历史时间序列数据的模式，并基于这些模式预测未来的数据点。RNN 是一种神经网络，可以记住先前输入的信息，这对于处理时间序列数据非常有用，因为时间序列数据的当前值往往与过去的值有关。
    
    DeepAR 还提供了一种处理缺失值和附加时间序列特征的方法。这使得它可以更好地处理现实世界的复杂时间序列问题，因为这些问题通常会包含缺失的数据点或者有额外的信息可以帮助进行预测。
    

## Some Concept

- SPEI
    
    标准化蒸发降水指数（Standardized Precipitation Evapotranspiration Index，简称SPEI）是一种用于评估干旱程度的指标。SPEI综合考虑了降水和蒸发两个因素，因此它不仅能反映降水的变化，还能反映温度和其他气候因素对蒸发的影响，从而更全面地评估干旱的程度。
    
    SPEI的计算方法如下：
    
    1. 首先，计算每个时间段（例如每月或每年）的蒸发降水差（即降水量减去蒸发量）。
    2. 然后，将蒸发降水差转换为概率分布，通常使用Gamma分布。
    3. 最后，将概率分布转换为标准正态分布，得到SPEI值。
    
    SPEI值的大小表示干旱的程度：正值表示湿润条件，负值表示干旱条件，值的绝对大小表示湿润或干旱的程度。例如，SPEI为-2表示严重干旱，SPEI为2表示严重湿润。
    
    SPEI可以计算不同的时间尺度，例如1个月，3个月，6个月，12个月等，以反映不同时间尺度上的干旱情况。例如，SPEI-3表示基于过去3个月的蒸发降水差计算的SPEI，它可以反映短期的干旱情况
    
- PACF
    
    偏自相关函数（Partial Autocorrelation Function，简称PACF）是时间序列分析中的一个重要概念。它用于描述一个时间序列中的观测值与其过去的观测值之间的关系，但是这种关系是在控制了其他过去观测值的影响之后的。
    
    具体来说，对于一个时间序列，其t时刻的观测值可能不仅与t-1时刻的观测值有关，还可能与t-2、t-3等更早的观测值有关。自相关函数（ACF）可以描述出t时刻的观测值与t-1、t-2、t-3等时刻的观测值之间的关系，但是这种关系中可能包含了一些间接的、通过其他过去观测值传递的影响。而偏自相关函数（PACF）则是在排除了这些间接影响之后，描述t时刻的观测值与t-1、t-2、t-3等时刻的观测值之间的直接关系。
    
    在实际应用中，PACF常常用于ARIMA模型等时间序列模型的建立。通过观察PACF的图形，可以帮助我们确定模型中的自回归项的阶数。例如，如果PACF在滞后k之后截尾（即从滞后k+1开始，所有的偏自相关系数都接近于0），那么我们可以考虑建立一个阶数为k的自回归模型。
    

## 数据处理pipeline

### Evaluation of Time Series Models in Simulating Different Monthly Scales of Drought Index for Improving Their Forecast Accuracy

1. **数据和研究区域**：文章首先选择了伊朗塞姆南市的气象数据作为研究对象，这些数据包括了从1973年到2020年的降雨统计数据。
2. **标准化降水指数（SPI）**：文章使用了标准化降水指数（SPI）来度量干旱。SPI是通过比较观测到的降水量和长期平均降水量来计算的。文章中详细解释了如何计算SPI，并给出了相关的公式。
3. **时间序列模型**：文章使用了ARIMA模型和它的一般形式，乘法季节性ARIMA，来预测干旱指数。文章中详细解释了如何选择最佳的模型，并给出了相关的公式。
4. **模型评估**：文章使用了不同的评估标准，如相关系数（R），均方根误差（RMSE），和平均绝对误差（MAE）来评估和分析模型的结果。
5. **模型预测**：在选择了最佳的模型后，文章使用了这个模型来预测未来24个月的干旱指数。文章中给出了预测结果的图表，并对结果进行了分析。
6. **观测和预测的干旱特性比较**：文章还比较了观测到的和预测的干旱特性，包括干旱的严重性，持续时间，和强度。
7. **结论**：文章的结论部分总结了研究的主要发现，并提出了对未来研究的建议。

### Estimation of SPEI Meteorological Drought using Machine Learning Algorithms

这篇文章的整体研究流程（pipeline）如下：

1. **数据获取和处理**：从中国国家气象数据共享平台获取气候变量数据，包括每日降水量、温度、太阳辐射、日照小时数、2米处的风速和相对湿度等。然后，使用Penman-Monteith（PM）方程计算参考蒸发散热（ET0）。接着，开发不同时间尺度的SPEI指数，通过n月的移动和来聚合月气候水平衡系列。
2. **SPEI计算**：计算月水平衡（Di），这是月降水量（Pi）和月潜在蒸发散热（PETi）之间的差值。然后，这些值在感兴趣的时间尺度上进行聚合。使用三参数对数逻辑概率分布来拟合D系列。
3. **机器学习模型训练和预测**：使用四种机器学习模型（XGB，RF，LSTM和CNN）来估计3个月和6个月时间尺度的SPEI。这些模型的训练和预测过程包括数据预处理、模型训练、模型优化和预测等步骤。
4. **干旱特性分析**：分析干旱的特性，包括总干旱持续时间（TDD）、干旱严重性（DS）、干旱峰值（DP）和干旱的空间范围（SEoD）。
5. **主成分分析（PCA）**：使用主成分分析（PCA）来识别干旱共变性的子区域模式。PCA可以区域化干旱的空间模式，反映了青藏高原气候的高时间变异性。

### Prediction of Impending Drought Scenarios Based on Surface and Subsurface Parameters in a Selected Region of Tropical Queensland, Australia

这篇文章是利用的ARIMA与ANN

1. **数据收集**：收集各种水文、气象和地下水参数，包括降水数据、标准降水指数（SPI）、土壤湿度指数（SMI）等。
2. **模型选择和训练**：选择并训练适合的模型，包括ARIMA模型和人工神经网络（ANN）模型。这些模型可以通过计算机编程自动生成。
3. **模型预测**：使用训练好的模型进行预测。例如，预测未来几个月的干旱指数，预测地表和地下水储量水平等。
4. **结果分析**：对预测结果进行分析，比较预测值和实际值，评估模型的准确性。如果预测结果不理想，可能需要考虑更多的参数，或者使用更大的数据集进行训练。
5. **模型优化**：根据结果分析，对模型进行优化。例如，如果发现预测结果有较大的误差，可能需要调整模型的参数，或者尝试使用其他的模型。
6. **模型应用**：将优化后的模型应用于实际问题，例如干旱的预测和管理。

### Monthly drought prediction based on ensemble models

在这篇文章中对与气象数据的处理可以借鉴，可以将数据集简化。这个过程的目的是将多个气候指数简化为一个单一的指数，以便更容易地进行干旱预测。这个单一的指数（ICP）包含了所有气候指数的信息，但是通过线性回归模型的权重分配，更重要的气候指数会对ICP有更大的影响。

1. **选择气候指数并简化为综合气候预测指数（ICP）**：首先，研究人员选择了一些气候指数，这些指数是关于气候条件的数据，如降雨量、温度等。然后，他们使用了一种叫做线性回归的统计方法，将这些指数简化为一个单一的指数，称为综合气候预测指数（ICP）。
    1. **选择气候指数**：研究人员选择了与目标月份降水量强相关的气候指数。这些气候指数可能包括海洋表面温度、大气压力等多个因素。
    2. **通过线性回归模型分配权重**：研究人员使用线性回归模型为这些气候指数分配权重。线性回归模型是一种统计学上的预测分析方法，可以用来量化两个或多个变量之间的关系。在这个过程中，每个气候指数都被赋予一个权重，这个权重反映了该指数对目标变量（即预测的降水量）的影响程度。
    3. **简化为综合气候预测指数（ICP）**：通过线性回归模型，研究人员将这些气候指数简化为一个单一的综合气候预测指数（ICP）。这个ICP是通过将每个气候指数乘以其相应的权重，然后将结果相加得到的。
2. **加权集合干旱预测（WEDP）模型的权重分配**：在这个步骤中，研究人员使用了一种方法来确定每个集合成员（也就是每个预测模型）在最终预测中的重要性。他们使用了两种方法来分配这些权重：传统权重（TW）和加权自助重采样（WBR）。
3. **条件集合干旱预测（CEDP）模型的边缘分布拟合**：在这个步骤中，研究人员使用了一种叫做对数正态分布的统计模型，来描述观测的ICP和P4;5;6（这是另一种气候指数）的分布。他们使用了一种叫做最大似然估计（MLE）的方法来估计这个模型的参数。
4. **生成条件集合干旱成员**：在这个步骤中，研究人员使用了适合的概率分布的累积分布函数（CDF）来生成1000个条件集合干旱成员。然后，他们取这1000个成员的平均值，得到了最终的干旱预测（CEDP）。
5. **测试预测的准确性和不确定性**：最后，研究人员使用了一些度量（如MSE、RMSE、MAE和NRMSE）来测试他们的预测有多准确，以及预测的不确定性有多大。

### A novel intelligent deep learning predictive model for meteorological drought forecasting

本文整体思路为

1. **数据收集**：研究人员收集了土耳其安卡拉省两个气象站（Beypazari和Nallihan）的气象数据，包括降水和温度时间序列。
2. **计算SPEI**：使用这些数据，研究人员计算了三个月和六个月的标准化蒸发降水指数（SPEI-3和SPEI-6）。SPEI是一个综合考虑了降水和蒸发的干旱指数。
3. **确定输入向量**：研究人员使用自相关函数（ACF）和偏自相关函数（PACF）分析确定了最佳的输入向量。ACF和PACF是时间序列分析中的工具，用于识别数据中的模式和结构。
4. **模型训练**：研究人员使用了一种名为CNN-LSTM的深度学习模型进行训练。这种模型结合了卷积神经网络（CNN）和长短期记忆网络（LSTM）的优点。CNN部分用于提取数据中的局部特征，而LSTM部分用于处理长距离的依赖关系。
    1. **卷积神经网络（CNN）**：CNN是一种深度学习模型，主要用于处理图像数据。CNN通过卷积层可以有效地提取局部特征，这对于处理时间序列数据也是非常有用的。在这个模型中，CNN部分被用作特征提取器，用于从原始数据中提取有用的特征。
    2. **长短期记忆网络（LSTM）**：LSTM是一种递归神经网络（RNN），它能够处理时间序列数据，捕捉长期依赖性。LSTM通过引入“门”结构，解决了传统RNN在处理长序列时容易出现的梯度消失和梯度爆炸问题。在这个模型中，LSTM部分被用于处理时间依赖性，用于从提取的特征中预测未来的干旱情况。
    
    这个模型被用于预测多时段干旱指数，特别是在土耳其安卡拉省的两个案例研究点上的三个月和六个月标准化蒸发降水指数（SPEI-3和SPEI-6）。通过统计精度度量、图形检查以及与基准模型（包括遗传编程、人工神经网络、LSTM和CNN）的比较来验证模型的效率。
    
5. **模型验证**：研究人员使用了统计精度度量、图形检查和与基准模型（包括遗传编程、人工神经网络、LSTM和CNN）的比较来验证所提出模型的效率。
6. **结果分析**：研究人员分析了模型的预测结果，并与其他模型进行了比较。结果显示，CNN-LSTM在所有基准测试中表现最好。

## 一些有关时间序列的包

1. tsfresh

[tsfresh — tsfresh 0.20.0.post0.dev20+ge93c796 documentation](https://tsfresh.readthedocs.io/)

*"Time Series Feature extraction based on scalable hypothesis tests"*.

The package provides systematic time-series feature extraction by combining established algorithms from statistics, time-series analysis, signal processing, and nonlinear dynamics with a robust feature selection algorithm. In this context, the term *time-series* is interpreted in the broadest possible sense, such that any types of sampled data or even event sequences can be characterised.

1. tsai

Time series Timeseries Deep Learning Machine Learning Pytorch fastai | State-of-the-art Deep Learning library for Time Series and Sequences in Pytorch / fastai

[https://github.com/timeseriesAI/tsai](https://github.com/timeseriesAI/tsai)

1. featuretools

[Featuretools](https://www.featuretools.com/)是一个用于自动化特征工程的python库

[https://github.com/alteryx/featuretools](https://github.com/alteryx/featuretools)

1. etna

ETNA是一个易于使用的时间序列预测框架。 它包括用于时间序列预处理、特征生成、 具有统一接口的各种预测模型 - 来自经典机器学习 到SOTA神经网络，模型组合方法和智能回溯测试。 ETNA旨在使时间序列的工作变得简单，高效和有趣。

[https://github.com/tinkoff-ai/etna](https://github.com/tinkoff-ai/etna)

1. darts

**Darts**是一个Python库，用于用户友好的预测和异常检测 在时间序列上。它包含多种型号，从经典的如ARIMA到 深度神经网络。预测模型都可以以相同的方式使用， 使用和函数，类似于scikit-learn。 该库还使回测模型变得容易， 结合多个模型的预测，并考虑外部数据。 Darts支持单变量和多变量时间序列和模型。 基于 ML 的模型可以在包含多个时间的潜在大型数据集上进行训练 系列，并且某些模型为概率预测提供了丰富的支持。

[https://github.com/unit8co/darts](https://github.com/unit8co/darts)

1. tslearn

用于 Python 中时间序列分析的机器学习工具包

[https://github.com/tslearn-team/tslearn](https://github.com/tslearn-team/tslearn)

1. sktime

具有时间序列的机器学习统一框架

[https://github.com/sktime/sktime](https://github.com/sktime/sktime)

1. pytorch-forcasting

PyTorch 预测是一个基于 *PyTorch* 的软件包，用于预测具有最先进的网络架构的时间序列。它提供了一个高级API，用于在熊猫数据帧上训练网络，并利用[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)在（多个）GPU，CPU上进行可扩展的训练和自动日志记录。

[https://github.com/jdb78/pytorch-forecasting](https://github.com/jdb78/pytorch-forecasting)
