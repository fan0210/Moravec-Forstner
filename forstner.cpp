#include "forstner.h"

namespace fd
{
	inline double ForstnerFeatureDetector::tr(const cv::Mat_<double> &m)const
	{
		double resV = 0;
		for (auto r = 0; r < m.rows; ++r)
		{
			for (auto c = 0; c < m.cols; ++c)
				if (r == c)
					resV += m.at<double>(r, c);
		}

		return resV;
	}

	void ForstnerFeatureDetector::detectCandidatePoints(const cv::Mat &img)
	{
		cv::Mat_<Robert> Guv(cv::Size(img.cols, img.rows));
		for (auto r = 0; r < img.rows - 1; ++r)
		{
			for (auto c = 0; c < img.cols - 1; ++c)
				Guv.at<Robert>(r, c) = Robert(img.at<uint_8>(r + 1, c + 1) - img.at<uint_8>(r, c), img.at<uint_8>(r, c + 1) - img.at<uint_8>(r + 1, c));
		}

		if (m_isSetThreshByWeight)
		{
			struct InterestValue
			{
				InterestValue() = default;
				InterestValue(const double &w, const double &q, const cv::Point &pt) :w(w), q(q), pt(pt) {}

				cv::Point pt;
				double w = 0;
				double q = 0;
			};

			std::vector<InterestValue> values;
			size_t w_total = 0;
			for (auto r = 0; r < Guv.rows - m_cellSize; ++r)
			{
				for (auto c = 0; c < Guv.cols - m_cellSize; ++c)
				{
					double gu2 = 0, gugv = 0, gv2 = 0;
					for (int i = 0; i < m_cellSize; ++i)
					{
						for (int j = 0; j < m_cellSize; ++j)
						{
							gu2 += pow(Guv.at<Robert>(r + i, c + j).gu, 2);
							gugv += Guv.at<Robert>(r + i, c + j).gu*Guv.at<Robert>(r + i, c + j).gv;
							gv2 += pow(Guv.at<Robert>(r + i, c + j).gv, 2);
						}
					}

					double trN = gu2 + gv2;

					if (trN != 0)
					{
						double detN = gu2*gv2 - gugv*gugv;

						double w = detN / trN;
						double q = 4 * detN / (trN*trN);

						values.push_back(InterestValue(w, q, cv::Point(c + m_cellSize / 2, r + m_cellSize / 2)));

						w_total += w;
					}
				}
			}

			if (m_isSetThreshByWeightMedian)
			{
				size_t size = values.size();

				std::unique_ptr<double[]> ws(new double[size]);

				size_t i = 0;
				for (auto it : values)
				{
					ws[i] = it.w;
					++i;
				}

				std::sort(ws.get(), ws.get() + size);

				if (size % 2 == 1)
					m_w_median = ws[size / 2];
				else
					m_w_median = (ws[size / 2 - 1] + ws[size / 2]) / 2.0;

				for (auto it = values.cbegin(); it != values.cend(); ++it)
					if (it->w > m_c*m_w_median&&it->q > m_thresh_q)
					{
						KeyPoint point = {
							it->pt,
							it->pt.y*img.cols + it->pt.x + 1,
							{it->w,it->q}
						};
						m_candidatePts.push_back(point);
					}
			}
			else
			{
				m_w_average = w_total / values.size();
				
				for (auto it = values.cbegin(); it != values.cend(); ++it)
				{
					if (it->w > m_f*m_w_average&&it->q > m_thresh_q)
					{
						KeyPoint point = {
							it->pt,
							it->pt.y*img.cols + it->pt.x + 1,
							{ it->w,it->q }
						};
						m_candidatePts.push_back(point);
					}
				}
			}
		}
		else
		{
			for (auto r = 0; r < Guv.rows - m_cellSize; ++r)
			{
				for (auto c = 0; c < Guv.cols - m_cellSize; ++c)
				{
					double gu2 = 0, gugv = 0, gv2 = 0;
					for (int i = 0; i < m_cellSize; ++i)
					{
						for (int j = 0; j < m_cellSize; ++j)
						{
							gu2 += pow(Guv.at<Robert>(r + i, c + j).gu, 2);
							gugv += Guv.at<Robert>(r + i, c + j).gu*Guv.at<Robert>(r + i, c + j).gv;
							gv2 += pow(Guv.at<Robert>(r + i, c + j).gv, 2);
						}
					}

					double trN = gu2 + gv2;

					if (trN != 0)
					{
						double detN = gu2*gv2 - gugv*gugv;

						double w = detN / trN;
						double q = 4 * detN / (trN*trN);

						if (q > m_thresh_q&&w > m_thresh_w)
						{
							KeyPoint point = {
								cv::Point2f(c + m_cellSize / 2,r + m_cellSize / 2),
								(r + m_cellSize / 2)*Guv.cols + (c + m_cellSize / 2) + 1,
								{ w,q }
							};
							m_candidatePts.push_back(point);
						}
					}
				}
			}
		}
	}

	void ForstnerFeatureDetector::filterKeyPoints(const cv::Mat &img)
	{
		for (auto r = 0; r < img.rows - m_blockSize + 1; r += m_blockSize)
		{
			for (auto c = 0; c < img.cols - m_blockSize + 1; c += m_blockSize)
			{
				KeyPoint point = { cv::Point2f(0,0),0,{ -1 ,-1} };

				for (auto it = m_candidatePts.cbegin(); it != m_candidatePts.cend(); ++it)
					if (isPointInBlock(*it, c, r))
						if (it->info[0] > point.info[0])
							point = *it;

				if (point.info[1] != -1)
					m_pts.push_back(point);
			}
		}

		m_candidatePts.clear();
	}

	inline bool ForstnerFeatureDetector::isPointInBlock(const KeyPoint &point, const int &c, const int &r)const
	{
		if (point.point.x >= c&&point.point.x < c + m_blockSize&&point.point.y >= r&&point.point.y < r + m_blockSize)
			return true;
		else
			return false;
	}

	const std::vector<ForstnerFeatureDetector::KeyPoint> &ForstnerFeatureDetector::detect(const cv::Mat &img)
	{
		if (img.channels() == 3)
		{
			cv::Mat grayImg;
			cv::cvtColor(img, grayImg, CV_BGR2GRAY);

			detectCandidatePoints(grayImg);
			std::cout << "提取到 " << m_candidatePts.size() << " 个候选点" << std::endl;
			filterKeyPoints(grayImg);
			std::cout << "过滤得到 " << m_pts.size() << " 个特征点" << std::endl;
		}
		else
		{
			detectCandidatePoints(img);
			std::cout << "提取到 " << m_candidatePts.size() << " 个候选点" << std::endl;
			filterKeyPoints(img);
			std::cout << "过滤得到 " << m_pts.size() << " 个特征点" << std::endl;
		}

		return m_pts;
	}
}