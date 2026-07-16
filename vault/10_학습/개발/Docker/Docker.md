---
유형: 주제노트
분야: 개발
주제: Docker
상태: 학습중
태그:
  - 개발/docker
---

# Docker([ˈdɒkər], 도커, 도커)

## 한 줄 요약
> 컨테이너로 개발 환경을 띄우고 관리. (주로 Supabase([ˈsuːpəbeɪs], 수파베이스, 수파베이스) 셀프호스팅 작업 메모)

## 자주 쓰는 명령어 / 스니펫
```
docker compose down && docker compose up -d
docker logs supabase-kong
docker ps | grep supabase-kong

curl -s -H "apikey: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJvbGUiOiJhbm9uIiwiZXhwIjoxOTgzODEyOTk2fQ.Hb8Mm_l3_Q1vHC7QsNZYsLDBjzn_Hy-jygzzuEjdU0k" http://localhost:8000/auth/v1/settings | jq
```

https://github.com/orgs/supabase/discussions/22359

```
# backup
supabase db dump --help
supabase projects list
supabase link --project-ref znnbrkrokalrzhtwuiqh
mkdir -p supabase-schema-backup
supabase db dump -f supabase-schema-backup/schema.sql
```

## 📌 암기 카드
> 앞면(cue)을 보고 뒷면을 떠올린 뒤 확인. (→ [[_핵심 암기장]])

| 앞면 (cue) | 뒷면 (외울 내용) |
|---|---|
| 컨테이너 재시작(전체) | `docker compose down && docker compose up -d` |
| 특정 컨테이너 로그 보기 | `docker logs <컨테이너명>` (예: `docker logs supabase-kong`) |
| 실행 중 컨테이너 필터 조회 | `docker ps \| grep <이름>` |
| Supabase 원격 프로젝트 연결 | `supabase link --project-ref <ref>` |
| Supabase 스키마 백업(dump) | `supabase db dump -f <파일경로>.sql` |