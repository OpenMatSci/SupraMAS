import json
import re
import sys
import time
import requests
import scipdf
import os
import aiohttp
import numpy as np
import random
from loguru import logger
from tenacity import retry, wait_fixed, stop_after_attempt
from supramas.protocol_optimizer.vector_retrieval import VectorRetrieval
import arxiv
from serpapi import GoogleSearch
import time
# Compatibility import: some scidownl builds expose only api.scihub.
# try:
#     from scidownl import scihub_download
# except ImportError:
from scidownl.api.scihub import scihub_download
import fitz  # PyMuPDF

def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type):
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            return target_str
        else:
            return text
    else:
        return ""


def fetch(url):
    try:
        timeout = 180
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/87.0.4280.88 Safari/537.36'
        } 
        with requests.Session() as session:
            with session.get(url, headers=headers, timeout=timeout, allow_redirects=True) as response:
                if response.status_code == 200:
                    content = response.content
                    return content
                else:
                    print(f"Failed to fetch the URL: {url} with status code: {response.status}")
                    return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching the URL: {url}", e)
        return None

    
class Result:
    def __init__(self,title="",abstract="",article=None,citations_conut = 0,year = None, pdf_url=None) -> None:
        self.title = title
        self.abstract = abstract
        self.article = article
        self.citations_conut = citations_conut
        self.year = year
        self.pdf_url = pdf_url


# Define the API endpoint URL

semantic_fields = ["title", "abstract", "year", "authors.name", "authors.paperCount", "authors.citationCount","authors.hIndex","url","referenceCount","citationCount","influentialCitationCount","isOpenAccess","openAccessPdf","fieldsOfStudy","s2FieldsOfStudy","embedding.specter_v1","embedding.specter_v2","publicationDate","citations"]


fieldsOfStudy = ["Computer Science","Medicine","Chemistry","Biology","Materials Science","Physics","Geology","Art","History","Geography","Sociology","Business","Political Science","Philosophy","Art","Literature","Music","Economics","Mathematics","Engineering","Environmental Science","Agricultural and Food Sciences","Education","Law","Linguistics"]



def process_fields(fields):
   return ",".join(fields)


class SementicSearcher:
    def __init__(
        self,
        ban_paper = [],
        grobid_url="http://172.16.8.29:8070",
        google_search_api_key=None,
        semantic_search_api_key=None,
        header_dir=None,
    ) -> None:
        self.ban_paper = ban_paper
        self.grobid_url = grobid_url
        self.google_search_api_key = google_search_api_key
        self.semantic_search_api_key = semantic_search_api_key
        self.header_dir = header_dir
        # 如果提供了header_dir，则创建nodes目录
        self.nodes_dir = os.path.join(self.header_dir, "nodes")
        os.makedirs(self.nodes_dir, exist_ok=True)


    @retry(wait=wait_fixed(10), stop=stop_after_attempt(5))
    def search_papers_(self, query, limit=10, offset=0, fields=["title", "paperId", "abstract", "isOpenAccess", 'openAccessPdf', "year","publicationDate","citations.title","citations.abstract","citations.isOpenAccess","citations.openAccessPdf","citations.citationCount","citationCount","citations.year"],
                            publicationDate=None, minCitationCount=0, year=None, 
                            publicationTypes=None, fieldsOfStudy=None):
        api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        headers = {"Accept": "application/json"}
        if self.semantic_search_api_key:
            headers["x-api-key"] = self.semantic_search_api_key

        params = {
            "query": query,
            "fields": ",".join(fields) if isinstance(fields, list) else fields,
            "offset": offset,
            "limit": limit,
        }
        if year is not None:
            params["year"] = year
        if publicationDate:
            params["publicationDateOrYear"] = publicationDate
        if minCitationCount:
            params["minCitationCount"] = minCitationCount
        if publicationTypes:
            params["publicationTypes"] = ",".join(publicationTypes) if isinstance(publicationTypes, list) else publicationTypes
        if fieldsOfStudy:
            params["fieldsOfStudy"] = ",".join(fieldsOfStudy) if isinstance(fieldsOfStudy, list) else fieldsOfStudy

        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.error(f"Semantic Scholar API error: HTTP {resp.status_code}, body={resp.text[:500]}")
                return None

            payload = resp.json()
            data = payload.get("data", []) or []
            total = payload.get("total", len(data))
            next_offset = offset + len(data) if data else offset
            return {"data": data, "total": total, "next": next_offset, "offset": offset}
        except Exception as e:
            logger.error(f"Semantic Scholar request failed: {e}")
            return None
            
    def cal_cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def cal_cosine_similarity_matric(self,matric1, matric2):
        if isinstance(matric1, list):
            matric1 = np.array(matric1)
        if isinstance(matric2, list):
            matric2 = np.array(matric2)
        if len(matric1.shape) == 1:
            matric1 = matric1.reshape(1, -1)
        if len(matric2.shape) == 1:
            matric2 = matric2.reshape(1, -1)
        dot_product = np.dot(matric1, matric2.T)
        norm1 = np.linalg.norm(matric1, axis=1)
        norm2 = np.linalg.norm(matric2, axis=1)

        cos_sim = dot_product / np.outer(norm1, norm2)
        scores = cos_sim.flatten()
        return scores.tolist()
    
    def rerank_papers(self, query_embedding, paper_list,llm):
        if len(paper_list) == 0:
            return []
        paper_list = [paper for paper in paper_list if paper]
        paper_contents = []
        for paper in paper_list:
            paper_content = f"""
                Title: {paper['title']}
                Abstract: {paper['abstract']}
                """
            paper_contents.append(paper_content)
        paper_contents_embbeding = llm.get_embbeding(paper_contents)
        paper_contents_embbeding = np.array(paper_contents_embbeding)
        scores = self.cal_cosine_similarity_matric(query_embedding,paper_contents_embbeding)
            
        # 根据score对paper_list进行排序 
        paper_list = sorted(zip(paper_list,scores),key = lambda x: x[1],reverse = True)
        paper_list = [paper[0] for paper in paper_list]
        return paper_list
    

    def search_google_scholar_papers_(self, query, limit=10, year=None, api_key=None):
        """
        使用 SerpAPI 的 Google Scholar 引擎检索论文信息
        :param query: 查询关键词
        :param limit: 最大返回数量
        :param year: 指定年份（可选）
        :param api_key: SerpAPI 的密钥
        :return: dict 包含 data、total、offset
        """
        print(f"Searching Google Scholar for query: <{query}>")

        params = {
            "engine": "google_scholar",
            "q": f"{query}",
            "api_key": api_key
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        if "error" in results:
            print(f"Google Scholar API error: {results['error']}")
        papers = results.get("organic_results", [])
        # print(papers)
        data = []
        for paper in papers:
            # print(paper)
            title = paper.get("title", "").strip()
            snippet = paper.get("snippet", "").strip()
            pub_info = paper.get("publication_info", {})
            pub_year = pub_info.get("year", None)
            year_text = paper.get("publication_info", {}).get("summary", "")
            match = re.search(r"\b(\d{4})\b", year_text)
            if match:
                pub_year_int = int(match.group(1))
            else:
                pub_year_int = None
            # print("year",pub_year_int)

            # 如果没有年份信息，跳过该论文
            # if pub_year_int is None or pub_year_int >= 2024:
            #     continue
            # if match and int(match.group(1)) < 2024:
            #     filtered_papers.append(paper)

            # 年份筛选
            # try:
            #     pub_year_int = int(pub_year) if pub_year else None
            # except:
            #     pub_year_int = None
            # if pub_year_int >= 2024:
            #     continue
            # if year and pub_year_int != year:
            #     continue

            # PDF 链接（如果存在）
            pdf_url = None
            for res in paper.get("resources", []):
                if "PDF" in res.get("file_format", ""):
                    pdf_url = res.get("link","")
                    # print(f"pdf_url: {pdf_url}")

            paper_dict = {
                "title": title,
                "abstract": snippet,
                "year": pub_year_int,
                "citationCount": 0,  # SerpAPI 无法获取引用数
                "isOpenAccess": pdf_url is not None,
                "openAccessPdf": {"url": pdf_url} if pdf_url else None,
                "pdf_url": pdf_url,
                "url": paper.get("link", "")
            }
            data.append(paper_dict)

            if len(data) >= limit:
                break

            # 限速防封（可选）
            time.sleep(0.5)

        return {"data": data, "total": len(data), "offset": 0}

    def search_arxiv_papers_(self, query, limit=10, sort_by="relevance", year=None):
        """
        使用 arxiv API 查询论文
        :param query: 查询关键词
        :param limit: 最大返回数量
        :param sort_by: 排序依据（"relevance", "lastUpdatedDate", "submittedDate"）
        :param year: 如果指定年份，将在筛选时应用
        :return: list[dict]
        """
        print(f"Searching ArXiv for query: <{query}>")

        # 查询
        search = arxiv.Search(
            query=query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance if sort_by == "relevance" else arxiv.SortCriterion.SubmittedDate
        )

        results = []
        for result in search.results():
            # 年份筛选（可选）
            pub_year = result.published.year
            if year and pub_year != year:
                continue
            print("result", result.pdf_url)
            paper_dict = {
                "title": result.title,
                "abstract": result.summary,
                "pdf_url": result.pdf_url,
                "published": result.published.strftime("%Y-%m-%d"),
                "year": pub_year,
                "citationCount": 0,  # ArXiv 不提供引用数
                "isOpenAccess": True,
                "openAccessPdf": {"url": result.pdf_url}
            }
            results.append(paper_dict)
            if len(results) >= limit:
                break

        return {"data": results, "total": len(results), "offset": 0}

    def search_semantic_open_access_papers_(self, query, limit=10, year=None, publicationDate=None, fields=None):
        """使用 Semantic Scholar API 搜索开源论文（仅保留可开放获取 PDF 的结果）"""
        if fields is None:
            fields = [
                "title",
                "paperId",
                "abstract",
                "isOpenAccess",
                "openAccessPdf",
                "year",
                "publicationDate",
                "citationCount",
                "url",
            ]

        # 适当放大召回，再做开源过滤
        raw_limit = max(limit * 3, 20)
        raw_results = self.search_papers_(
            query,
            limit=raw_limit,
            year=year,
            publicationDate=publicationDate,
            fields=fields,
        )

        if not raw_results or "data" not in raw_results:
            return {"data": [], "total": 0, "offset": 0}

        open_access_data = []
        for paper in raw_results["data"]:
            oa_pdf = paper.get("openAccessPdf") or {}
            oa_url = oa_pdf.get("url")
            if paper.get("isOpenAccess") and oa_url:
                # 保留一个统一字段，便于下游下载逻辑复用
                paper["pdf_url"] = oa_url
                open_access_data.append(paper)
                if len(open_access_data) >= limit:
                    break

        return {"data": open_access_data, "total": len(open_access_data), "offset": 0}

    def normalize_title(self, title: str):
        # 替换 / 和 \ 为一个空格
        title = re.sub(r'[\\/:*?"<>|]', " ", title)
        # title = re.sub(r"(\.\.\.|…)", " ", title)
        # 替换多个连续空格为一个下划线
        title = re.sub(r"\s+", "_", title.strip())
        return title


    def _filter_unique_results(self, results, ban_papers):
        """过滤重复和禁止的论文"""
        seen_titles = set()
        unique_results = []
        
        for result in results:
            title = result["title"]
            if title not in seen_titles and title not in ban_papers:
                seen_titles.add(title)
                unique_results.append(result)
        
        return unique_results

    def _check_paper_availability(self, result, read_papers):
        """检查论文是否可用（本地存在或开放获取）"""
        title = result["title"]
        normalized_title = self.normalize_title(title)
        
        if title in read_papers:
            return None, None  # 已读，跳过
        
        # 检查本地文件
        node_path = os.path.join(self.nodes_dir, f"{normalized_title}.pdf")
        
        if os.path.exists(node_path):
            result["url"] = node_path
            return result, None
        elif result.get("isOpenAccess") and result.get("openAccessPdf"):
            return result, None
        else:
            # 需要手动上传
            result["normalized_title"] = normalized_title
            return None, result

    def _wait_for_manual_upload(self, pending_result):
        """等待用户手动上传论文"""
        title = pending_result["title"]
        normalized_title = self.normalize_title(title)
        node_path = os.path.join(self.nodes_dir, f"{normalized_title}.pdf")
        
        print(f"\n[闭源论文] {title}")
        print(f"DOI/URL: {pending_result.get('url', '无')}")
        print(f"请手动下载并保存到nodes目录下(重命名一下:空格用_代替): {self.nodes_dir}/{normalized_title}.pdf")
        
        check_interval = 5
        while True:
            if os.path.exists(node_path):
                pending_result["url"] = node_path
                print(f"[✓] 已识别上传论文：{title}")
                return pending_result
            try:
                print("尚未检测到文件，按 Esc 退出，任意键继续...")
                key = input()
                if key == '\x1b':  # Esc 键（需终端支持）
                    print("已退出")
                    return None
            except Exception as e:
                print(f"An error occurred: {e}")
                return None

    def _process_paper_candidates(self, candidates, max_results):
        """处理论文候选列表，构建最终结果"""
        final_results = []
        
        for result in candidates:
            if len(final_results) >= max_results:
                break
                
            title = result["title"]
            abstract = result.get("abstract", "")
            citation_count = result.get("citationCount", 0)
            year = result.get("year", None)
            pdf_url = result.get("url", "")
            
            # print(f"pdf_url: {pdf_url}")
            final_results.append(Result(title, abstract, None, citation_count, year, pdf_url))
        
        return final_results

    def search_(self, query, max_results=5, paper_list=None, rerank_query=None, llm=None, year=None,
                publicationDate=None, need_download=True, fields=None):
        """搜索论文的主方法 - 重构版本"""
        if fields is None:
            fields = ["title", "paperId", "abstract", "isOpenAccess", 'openAccessPdf', "year", 
                    "publicationDate", "citationCount"]
        
        # # 2. 搜索论文（保留原有逻辑：Google Scholar）
        # scholar_results = self.search_google_scholar_papers_(
        #     query, limit=max_results, year=year, api_key=self.google_search_api_key
        # )

        # 备选实现（仅注释保留，不启用）：
        # 先走 Semantic Scholar API（仅开源）
        scholar_results = self.search_semantic_open_access_papers_(
            query,
            limit=max_results,
            year=year,
            publicationDate=publicationDate,
            fields=fields,
        )
        # 结果不足时，可回退到 Google Scholar
        # if not scholar_results or len(scholar_results.get("data", [])) == 0:
        #     scholar_results = self.search_google_scholar_papers_(
        #         query, limit=max_results, year=year, api_key=self.google_search_api_key
        #     )

        logger.info(f"搜到论文结果{len(scholar_results['data'])}篇")
        if not scholar_results or "data" not in scholar_results:
            return []
        
        # 3. 过滤重复和禁止的论文
        unique_results = self._filter_unique_results(scholar_results["data"], self.ban_paper)
        
        # 4. 处理论文可用性
        paper_candidates = []
        

        if need_download:
            for index, result in enumerate(unique_results):
                print(f"Processing paper {index + 1}/{len(unique_results)}: {result['title']}")
                if paper_candidates != []:
                    break
                
                # 规范title
                normalized_title = self.normalize_title(result["title"])

                node_path = os.path.join(self.nodes_dir, f"{normalized_title}.pdf")
                try:
                    if os.path.exists(node_path) and fitz.open(node_path):
                        result["url"] = node_path
                        paper_candidates.append(result)

                    else:   
                        # 直接下载论文
                        if result.get("pdf_url") is not None:
                            print(f"开始直接下载: {result.get('title')},URL: {result.get('url')}")
                            self.download_pdf_(result.get("pdf_url"), node_path)
                            if os.path.exists(node_path) and fitz.open(node_path):
                                result["url"] = node_path
                                paper_candidates.append(result)
                                print(f"直接从下载论文成功: {result.get('title')}")
                                continue
                            else:
                                print(f"直接下载论文失败: {result.get('title')}")

                        # # 通过第三方下载
                        # if result.get("title") is not None:
                        #     print(f"开始第三方下载: {result.get('title')},URL: {result.get('url')}")
                        #     scihub_download(result.get("title"),paper_type="title", out=node_path)
                        #     if os.path.exists(node_path) and fitz.open(node_path):
                        #         result["url"] = node_path
                        #         paper_candidates.append(result)
                        #         print(f"第三方下载论文成功: {result.get('title')}")
                        #         continue
                        #     else:
                        #         print(f"第三方下载论文失败: {result.get('title')}")


                        # 手动上传流程已禁用：若直接下载和第三方下载都失败，自动尝试下一篇
                        # print(f"开始等待手动上传: {result.get('title')}")
                        # uploaded_result = self._wait_for_manual_upload(result)
                        # if uploaded_result:
                        #     paper_candidates.append(uploaded_result)
                        continue

                except Exception as e:
                    print(f"Error processing paper {result.get('title')}: {e}")
                    # 手动上传流程已禁用：异常时自动尝试下一篇
                    continue
                
        else:
            # 不需要下载时，直接添加所有结果
            paper_candidates = unique_results
        
        # 5. 构建最终结果
        final_results = self._process_paper_candidates(paper_candidates, max_results)

        logger.info(f"已下载论文: {len(final_results)}篇")

        return final_results

    def search_related_paper_(self,title,need_citation = True,need_reference = True,rerank_query = None,llm = None,paper_list = []):
        print(f"Searching for related papers of paper <{title}>; Citation:{need_citation}; Reference:{need_reference}")
        fileds = ["title","abstract","citations.title","citations.abstract","citations.citationCount","references.title","references.abstract","references.citationCount","citations.isOpenAccess","citations.openAccessPdf","references.isOpenAccess","references.openAccessPdf","citations.year","references.year"]
        results = self.search_papers_(title,limit = 30,fields=fileds)
        related_papers = []
        related_papers_title = []
        if not results or "data" not in results:
            print(f"Failed to find related papers of paper <{title}>; Citation:{need_citation}; Reference:{need_reference}")
            return None
        for result in results["data"]:
            if not result:
                continue
            if need_citation:
                for citation in result["citations"]:
                    if os.path.exists(os.path.join(self.save_file, f"{citation['title']}.pdf")) and citation["title"] not in paper_list:
                        if "openAccessPdf" not in citation or not citation["openAccessPdf"] or "url" not in citation["openAccessPdf"]:
                            citation["openAccessPdf"] = {"url":None}
                        related_papers.append(citation)
                        related_papers_title.append(citation["title"])
                    elif citation["title"] in related_papers_title or citation["title"] in self.ban_paper or citation["title"] in paper_list:
                        continue
                    elif citation["isOpenAccess"] == False or citation["openAccessPdf"] == None:
                        continue
                    else:
                        related_papers.append(citation)
                        related_papers_title.append(citation["title"])
            if need_reference:
                for reference in result["references"]:
                    if os.path.exists(os.path.join(self.save_file, f"{reference['title']}.pdf")) and reference["title"] not in paper_list:
                        if "openAccessPdf" not in reference or not reference["openAccessPdf"] or "url" not in reference["openAccessPdf"]:
                            reference["openAccessPdf"] = {"url":None}
                        related_papers.append(reference)
                        related_papers_title.append(reference["title"])
                    elif reference["title"] in related_papers_title or reference["title"] in self.ban_paper or reference["title"] in paper_list:
                        continue
                    elif reference["isOpenAccess"] == False or reference["openAccessPdf"] == None:
                        continue
                    else:
                        related_papers.append(reference)
                        related_papers_title.append(reference["title"])
            if result:
                break
        
        if len(related_papers) >= 20:
            related_papers = random.sample(related_papers,20)

        if rerank_query and llm:
            rerank_query_embbeding = llm.get_embbeding(rerank_query)
            rerank_query_embbeding = np.array(rerank_query_embbeding)
            related_papers = self.rerank_papers(rerank_query_embbeding, related_papers,llm)
            related_papers = [[paper["title"],paper["abstract"],paper["openAccessPdf"]["url"],paper["citationCount"],paper['year']] for paper in related_papers]
        else:
            related_papers = [[paper["title"],paper["abstract"],paper["openAccessPdf"]["url"],paper["citationCount"],paper['year']] for paper in related_papers]
            related_papers = sorted(related_papers,key = lambda x: x[3],reverse = True)
        print(f"Found {len(related_papers)} related papers")
        for paper in related_papers:
            url = paper[2]
            article = self.read_arxiv_from_link_(url, f"{paper[0]}.pdf")
            if not article:
                continue
            result = Result(paper[0],paper[1],article,paper[3],paper[4])
            return result
        print(f"Failed to find related papers of paper <{title}>; Citation:{need_citation}; Reference:{need_reference}")
        return None

    

    def read_arxiv_from_link_(self, pdf_link , filename):
        # 优先检查nodes目录
        node_file_path = os.path.join(self.nodes_dir, filename)
        if os.path.exists(node_file_path):
            article_dict = self.read_arxiv_from_path(node_file_path)
            return article_dict
            
        # 下载到nodes目录
        result = self.download_pdf_(pdf_link, node_file_path)
        if not result:
            print(f"Failed to download the PDF file: {filename}")
            return None
        try:
            article_dict = self.read_arxiv_from_path(file_path)
            return article_dict
        except Exception as e:
            print(f"Failed to read the article from the PDF file: {e}, {filename}")
            return None

    def read_arxiv_from_path(self, pdf_path):
        if not os.path.exists(pdf_path):
            print(f"The PDF file <{pdf_path}> does not exist.")
            return None
        try:
            article_dict = scipdf.parse_pdf_to_dict(pdf_path, grobid_url=self.grobid_url)
        except Exception as e:
            print(f"Failed to read the article from the PDF file: {e}, {pdf_path}")
            return None
        return article_dict


    def download_pdf_(self, pdf_link, save_path):
        if os.path.exists(save_path):
            print(f"The PDF file <{save_path}> already exists.")
            return True
        content = fetch(pdf_link)
        if not content:
            print(f"Failed to download the PDF: {save_path}")
            return False
        try:
            with open(save_path, 'wb') as file:
                file.write(content)
            return True
        except Exception as e:
            print(f"Failed to download the PDF file: {e}, {save_path}")
            return False


    def read_paper_title_abstract(self,article):
        title = article["title"]
        abstract = article["abstract"]
        paper_content = f"""
Title: {title}
Abstract: {abstract}
        """
        return paper_content
    

    def read_paper_title_abstract_introduction(self,article):
        title = article["title"]
        abstract = article["abstract"]
        introduction = article["sections"][0]["text"]
        paper_content = f"""
Title: {title}
Abstract: {abstract}
Introduction: {introduction}
        """
        return paper_content

    def read_paper_content(self, article):
        paper_content = self.read_paper_title_abstract(article)
        for section in article["sections"]:
            # print("DEBUG: section =", section)
            # print("DEBUG: section.keys() =", section.keys())

            # print("section['publication_ref']", section["publication_ref"])
            paper_content += (f"section: {section['heading']}\n content: {section['text']}\n "
                              f"ref_ids: {section['n_publication_ref']}\n")
        return paper_content

    def read_paper_content_with_ref(self, article):
        paper_content = self.read_paper_content(article)
        # print("DEBUG: paper_content =", paper_content)
        paper_content += "<References>\n"
        for refer in article["references"]:
            # print("DEBUG: refer =", refer)
            # ref_id = refer["ref_id"]
            title = refer["title"]
            year = refer["year"]
            paper_content += f"Title: {title} Year: ({year})\n"
        paper_content += "</References>\n"
        return paper_content

        
