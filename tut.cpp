/* #include <bits/stdc++.h>
using namespace std;
#define ll long long

void solve(vector<int> &nums, int index, vector<int> &output, vector<vector<int>> &ans, ll sum, int number, int remove)
{
   if (index >= nums.size())
   {
       if (sum % number == 0 && output.size() != 0)
       {
           ans.push_back(output);
       }
       return;
   }
   if (index == remove)
   {
       solve(nums, index + 1, output, ans, sum, number, remove);
   }
   else
   {
       // exclude
       solve(nums, index + 1, output, ans, sum, number, remove);
       // inculde
       sum += nums[index];
       output.push_back(nums[index]);
       solve(nums, index + 1, output, ans, sum, number, remove);
       output.pop_back();
       sum -= nums[index];
   }
}

int main()
{
   ios_base::sync_with_stdio(false);
   cin.tie(NULL);
   int t;
   cin >> t;
   while (t--)
   {
       int n;
       cin >> n;
       int m;
       cin >> m;
       vector<int> arr;
       for (int i = 0; i < n; i++)
       {
           int ele;
           cin >> ele;
           arr.push_back(ele);
       }
       vector<int> temp;
       vector<vector<int>> output;
       int index = 0;
       bool ans = true;
       for (int i = 0; i < n; i++)
       {
           solve(arr, index, temp, output, 0, m, i);
           if (output.empty())
           {
               cout << "NO" << endl;
               ans = false;
               break;
           }
           temp.clear();
           output.clear();
       }
       if (ans == true)
       {
           cout << "YES" << endl;
       }
   }
   return 0;
}  */
/*
#include <bits/stdc++.h>
using namespace std;

int main()
{   unordered_map<int,int> voldemort_vs_wizard;
    int n;
    cin >> n;
    int b;
    cin >> b;
    vector<int> harry;
    vector<int> voldemort_army;
    for (int i = 0; i < n; i++)
    {
        int element;
        cin >> element;
        harry.push_back(element);
    }
    for (int i = 0; i < b; i++)
    {
        int volemort;
        cin >> volemort;
        int wizard;
        cin >> wizard;
        voldemort_army.push_back(volemort);
        voldemort_vs_wizard[volemort]=wizard;
    }
    sort(voldemort_army.begin(), voldemort_army.end());
    int local_sum = 0;


    return 0;
} */
/*
#include <bits/stdc++.h>
using namespace std;
void implement_fun(int n, int m, vector<int> &arr, int sum, int index, int *flag, vector<int> &output)
{
    if (index >= n-1) //1
    {
        if (sum % m == 0 && output.size()!=0)
        {
            // cout<<sum<<"sum"<<endl;
            *flag = 1;
        }
        return;
    }
    //ex
    implement_fun(n, m, arr, sum, index + 1, flag, output);
    int ch = arr[index];
    output.push_back(ch);
    sum += ch;
    implement_fun(n, m, arr, sum, index + 1, flag, output);
    output.pop_back();
    sum -= ch;
}
int main()
{
    int testcase;
    cin >> testcase;
    int n, m;
    bool ans;
    for (int k = 0; k < testcase; k++)
    {
        cin >> n;
        cin >> m;
        vector<int> arr;
        vector<int> arr2; // declaring a element of n-1 size
        int count = 0;
        int flag2 = 0; // to check whether all the array have the subsequence divisible by m
        // Taking input for the array
        for (int i = 0; i < n; i++)
        {
            int ele;
            cin >> ele;
            arr.push_back(ele);
        }
        if (n == 1)
        {
            flag2 = 0;
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                arr2.clear();
                int flag = 0;
                for (int j = 0; j < n; j++)
                {
                    if (count != j)
                    {
                        arr2.push_back(arr[j]);
                    }
                }
                // for(int i = 0; i<n-1; i++){
                //     cout<<arr2[i]<<" ";
                // }
                // cout<<endl;
                int sum = 0;
                int index = 0;
                vector<int> output;
                implement_fun(n, m, arr2, sum, index, &flag, output); // checking all subsequesnce is divisible by m
                if (flag == 1)
                { // if flag ==1 means at least one susequence divisible by m
                    flag2 = 1;
                    // cout<<"miil gya"<<endl;
                }
                else
                {
                    flag2 = 0;
                    // cout<<"are yr ho kyu nhi rha"<<endl;
                    break;
                }
                count++;
            }
        }
        if (flag2 == 1)
        {
            cout << "YES" << endl;
        }
        else
        {
            cout << "NO" << endl;
        }
    }
} */
/*
#include <bits/stdc++.h>
using namespace std;
#define ll long long
int save_wizard(vector<int> &voldemort_army, int attack)
{
    int low = 0;
    int high = voldemort_army.size() - 1;
    int mid = low + (high - low) / 2;
    int last_val = -1;
    while (low <= high)
    {
        if (voldemort_army[mid] <= attack)
        {
            low = mid + 1;
            last_val = mid;
        }
        else
        {
            high = mid - 1;
        }
        mid = (low + high) / 2;
    }
    return last_val;
}

int main()
{
    unordered_map<int, int> voldemort_vs_wizard;
    int n;
    cin >> n;
    int b;
    cin >> b;
    vector<int> harry_army;
    for (int i = 0; i < n; i++)
    {
        int attack;
        cin >> attack;
        harry_army.push_back(attack);
    }
    vector<int> voldemort_army;
    for (int i = 0; i < b; i++)
    {
        int voldemort;
        cin >> voldemort;
        int wizard;
        cin >> wizard;
        voldemort_army.push_back(voldemort);
        auto it = voldemort_vs_wizard.find(voldemort);
        if (it != voldemort_vs_wizard.end())
        {
            it->second += wizard;
        }
        else
        {
            voldemort_vs_wizard[voldemort] = wizard;
        }
    }
    sort(voldemort_army.begin(), voldemort_army.end());
    vector<int> hashed_array;
    ll sum = 0;
    int prev = 0;
    int prev_vol=-1;
    for (int i = 0; i < b; i++)
    {
        int vol = voldemort_army[i];
        int value = voldemort_vs_wizard[vol];
        int ans;
        if(vol==prev_vol){
             ans = prev;
        }
        else{
         ans = prev + value;
        }
        hashed_array.push_back(ans);
        prev = ans;
        prev_vol=vol;
    }
    for (int i = 0; i < n; i++)
    {
        int max_voldemort = save_wizard(voldemort_army, harry_army[i]);
        if (max_voldemort == -1)
        {
            // cant defeat anyone
            sum = 0;
        }
        else
        {
            sum = hashed_array[max_voldemort];
        }
        cout << sum << " ";
    }

    return 0;
} */

/*
#include <bits/stdc++.h>
using namespace std;
#define ll long long

int same_kr_char_ko(string &input,int start,int end,char ch){
    int result=0;
    for(int i=start;i<=end;i++){
        if(ch!=input[i]){
            result++;
        }
    }
    return result;
}

int energy_drink_pi_lo(string &input, int start, int end, char comparison)
{
    if (end - start + 1 == 1)
    {
        // one char string
        if(input[start]!=comparison){
            return 1;
        }
        return 0;
    }
    // left part
    int mid = (start + end) / 2;

    int left_cost1=same_kr_char_ko(input,start,mid,comparison);//0 3
    int right_cost1=energy_drink_pi_lo(input,mid+1,end,comparison+1);
    int opt1=left_cost1+right_cost1;

    int left_cost2=energy_drink_pi_lo(input,start,mid,comparison+1);
    int right_cost2=same_kr_char_ko(input,mid+1,end,comparison);
    int op2=left_cost2+right_cost2;

    return min(opt1,op2);

}

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t;
    cin >> t;
    while (t--)
    {
        int n;
        cin >> n;
        string s;
        cin >> s;
        int start = 0;
        int high = s.length() - 1;
        int mid = (start + high) / 2;
        char campare = 'a';
        // 2 recursive calls start,mid  mid+1,high
        //same kr char ko in left
        int answer = energy_drink_pi_lo(s,start,high,campare);
        cout<<answer<<endl;

    }
    return 0;
}  */
/*
#include <bits/stdc++.h>
using namespace std;
bool cmp(pair<int, int> &a,
         pair<int, int> &b)
{
    return a.second < b.second;
}
void sort_map(map<int, int> &M)
{

    vector<pair<int, int>> A;
    for (auto &it : M)
    {
        A.push_back(it);
    }

    sort(A.begin(), A.end(), cmp);
}

int main()
{
    int n;
    int m;
    cin >> n >> m;
    vector<int> arr(n);
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }
    // mergeSort(arr, 0, n);
    map<int, int> m1;
    int a;
    int b;
    int sum = 0;
    for (int i = 0; i < m; i++)
    {
        cin >> a >> b;
        auto it = m1.find(a);
        if (it != m1.end())
        {
            it->second += b;
        }
        else
        {
            m1.insert({a, b});
        }
    }
    sort_map(m1);
    for (auto &a1 : m1)
    {
        sum += a1.second;
        a1.second = sum;
    }
    vector<int> ans;
    for (int i = 0; i < n; i++)
    {
        auto it = m1.lower_bound(arr[i]);
        if (it == m1.begin() && arr[i] < it->first)
        {
            ans.push_back(0);
           // cout << 0 << endl;
        }
        else if (it == m1.end())
        {
            it--;
            ans.push_back(it->second);
            //cout << it->first << endl;
        }
        else
        {
            if (it->first == arr[i])
            {
                ans.push_back(it->second);
                //cout << it->first << endl;
            }
            else
            {
                it--;
                ans.push_back(it->second);
                //cout << it->first << endl;
            }
        }
         if (it != m1.begin() && arr[i] < it->first)
        {
            --it;
        }
        if (it != m1.end())
        {
            if (arr[i] < it->first)
            {
                ans.push_back(0);
            }
            else
            {
                ans.push_back(it->second);
            }
        }
        else
        {
            ans.push_back(0);
        }
    }

    for (int i = 0; i < ans.size(); i++)
    {
        cout << ans[i] << " ";
    }
}
 */
/*
#include <iostream>
#include <vector>

using namespace std;

void divisible(const vector<int> &arr, int e, int i, int sum, int size, bool found,vector<int> &out,vector<int> store)
{

    if (i >=arr.size())
    {
        if(sum%e==0 && found!=false && store.size()!=0){
            out.push_back(sum);
            return ;
        }
        return ;
    }

    //exclude
    divisible(arr,e,i+1,sum,size,true,out,store);
    //include
    store.push_back(arr[i]);
    divisible(arr,e,i+1,sum+arr[i],size,true,out,store);

}

int main()
{
    int t;
    cin >> t;

    while (t--)
    {
        int size, e;
        cin >> size;
        cin >> e;

        vector<int> arr(size);
        for (int i = 0; i < size; i++)
        {
            cin >> arr[i];
        }
        vector<int> array;
        vector<int> output;
        bool f=true;
        for(int i=0;i<size;i++){
            array.clear();
            output.clear();
            int remove=i;
            if(size==1){
                cout<<"NO"<<endl;
                f=false;
                break;
            }
            for(int j=0;j<size;j++){
                if(remove!=j){
                    array.push_back(arr[j]);
                }
            }
            vector<int> store;
            divisible(array,e,0,0,size-1,true,output,store);
            if(output.size()==0){
                cout<<"NO"<<endl;
                f=false;
                break;
            }
        }
        if(f){
        cout<<"YES"<<endl;
        }
    }

    return 0;
} */

/*
#include <bits/stdc++.h>
using namespace std;
#define ll long long

void pikachu_m_tumhe_chunta_hu(vector<int> &ans,int start,int end,vector<int> array){
    int min = array[start];
    int max =array[end-1];
    int mid = (min+max)/2;
    //calculate sum for left and right
    int left=0;
    int right=0;
    for(int i=start;i<mid;i++){
        left+=array[i];
    }
    for(int i=mid;i<end;i++){
        right+=array[i];
    }
    cout<<left<<" "<<right<<" ";
    ans.push_back(left);
    ans.push_back(right);
    if(min==max){
        return;
    }
    if(start>=end){
        return;
    }
    pikachu_m_tumhe_chunta_hu(ans,start,mid,array);
    pikachu_m_tumhe_chunta_hu(ans,mid,end,array);
}


int binarySearch(vector<int> arr, int size, int key) {

    int start = 0;
    int end = size-1;

    int mid = start + (end-start)/2;

    while(start <= end) {

        if(arr[mid] == key) {
            return mid;
        }

        //go to right wala part
        if(key > arr[mid]) {
            start = mid + 1;
        }
        else{ //key < arr[mid]
            end = mid - 1;
        }

        mid = start + (end-start)/2;
    }

    return -1;
}

int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t; cin >> t;
    while(t--){
        int n; cin >> n;
        int q;cin>>q;
        vector<int> arr;
        for(int i=0;i<n;i++){
            int element;cin>>element;
            arr.push_back(element);
        }
        sort(arr.begin(),arr.end());
        vector<int> threesome;
        pikachu_m_tumhe_chunta_hu(threesome,0,n,arr);
        sort(threesome.begin(),threesome.end());
        for(int i=0;i<q;i++){
            int query;cin>>query;
            int result = binarySearch(threesome,threesome.size(),query);
            if(result!=-1){
                cout<<"YES"<<endl;
            }
            else{
                cout<<"NO"<<endl;
            }
        }
    }
    return 0;
} */

/*
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main() {
    int n, b;
    cin >> n >> b;

    vector<long> tPower(n);
    vector<pair<long, long>> bases(b);
    vector<long> originalOrder(n);

    for (int i = 0; i < n; i++) {
        cin >> tPower[i];
        originalOrder[i] = i;
    }

    for (int i = 0; i < b; i++) {
        cin >> bases[i].first >> bases[i].second;
    }

    sort(bases.begin(), bases.end());

    vector<long long> prefixSum(b + 1, 0);
    for (int i = 1; i <= b; i++) {
        prefixSum[i] = prefixSum[i - 1] + bases[i - 1].second;
    }

    auto binarySearch = [&](int target) {
        int left = 0, right = b;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (bases[mid].first <= target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    };

    vector<long long> wizardsSaved(n);
    for (int i = 0; i < n; i++) {
        int originalIndex = originalOrder[i];
        int expectedIndex = binarySearch(tPower[originalIndex]);
        wizardsSaved[originalIndex] = prefixSum[expectedIndex];

        if (tPower[originalIndex] == bases[0].first) {
            wizardsSaved[originalIndex] += bases[0].second;
        }
    }

    for (int i = 0; i < n; i++) {
        cout << wizardsSaved[i] << " ";
    }

    return 0;
} */
/*
#include <bits/stdc++.h>
using namespace std;
vector<int> solve(vector<int> &arr)
{
    vector<int> result;
    // 1st trivial case
    int maxi = *max_element(arr.begin(), arr.end());
    result.push_back(maxi);
    // nor for 2 to n
    int temp = 2; // subarray size
    vector<int> faltu_ka_vector;
    vector<int> faltu;
    while (temp < arr.size())
    {
        faltu_ka_vector.clear();
        for (int i = 0; i < arr.size(); i++)
        {
            faltu.clear();

            for (int j = i; j < i + temp && j < arr.size(); j++)
            {
                faltu.push_back(arr[j]);
            }

            if (i + temp <= arr.size())
            {
                int mini = *min_element(faltu.begin(), faltu.end());
                faltu_ka_vector.push_back(mini);
            }
        }
        cout << endl;
        int maxi = *max_element(faltu_ka_vector.begin(), faltu_ka_vector.end());
        result.push_back(maxi);
        temp++;
    }
    // trivial case;
    int mini = *min_element(arr.begin(), arr.end());
    result.push_back(mini);
    return result;
}

int main()
{
    vector<int> arr = {3,1,2,4};

    vector<int> result = solve(arr);

    cout << "Resulting array: ";
    for (int value : result)
    {
        cout << value << " ";
    }
    cout << endl;

    return 0;
}
 */
// 0 for ding
// 1 for ring
/*
#include <bits/stdc++.h>
using namespace std;
int solve(vector<int> a, int rcount, int dcount, int score, int pointer)
{
    if (pointer >= a.size())
    {
        return score;
    }
    // try to ding
    if(dp[pointer][0]!=INT_MIN || dp[pointer][1]!=INT_MIN){
        if(dp[pointer][0]!=INT_MIN){
            return dp[pointer][0];
        }
        else{
            return dp[pointer][1];
        }
    }
    int ding = 0;
    int ring = 0;
    if (dcount < 3)
    {
        int sc = 0;
        if (a[pointer] < 0)
        {
            sc = abs(a[pointer]);
        }
        else
        {
            sc = a[pointer] * -1;
        }
        ding = solve(a, 0, dcount + 1, score + sc, pointer + 1);
    }
    else
    {
        // ring always
        ring = solve(a, rcount + 1, 0, score + a[pointer], pointer + 1);
    }

    // try for ring
    if (rcount < 3)
    {
        int sc = 0;
        ring = solve(a, rcount + 1, 0, score + a[pointer], pointer + 1);
    }
    else
    {
        int sc = 0;
        if (a[pointer] < 0)
        {
            sc = abs(a[pointer]);
        }
        else
        {
            sc = a[pointer] * -1;
        }
        ding = solve(a, 0, dcount + 1, score + sc, pointer + 1);
    }
    score = max(ding, ring);
    dp[pointer][0]=score;
    dp[pointer][1]=score;
    return score;
}

int main()
{
    // Example usage
    vector<int> A = {10, 40, 20, -2, -5, -3, 4, 6, 8, 20};
    int dp[100000][2]; //a func which initailzes this DP array to INT_MIN
    int result = solve(A, 0, 0, 0, 0);

    cout << "The largest number of chickens earned by Mr. Fox is: " << result << endl;

    return 0;
}
 */

/*
#include<bits/stdc++.h>
using namespace std;

int solve( vector<vector<int>> &price,int width,int height){
    if(width==1 && height==1){
        return price[0][0];
    }
    int p=-1;

    //create cases
    while(width>0){
        //kuch kro
        //store kro abhi ka result
        int abhi=price[width-1][height-1];
        //ab dekho max of abhi nd recursive call ka result
        int new_p_part1=solve(price,width-1,height);
        int new_p_part2=solve(price,1,height);
        int result = max(abhi,new_p_part1+new_p_part2);
        p=max(p,result);
        width--;
    }
    while(height>0){
        //kuch kro
        //store kro abhi ka result
        int abhi=price[width-1][height-1];
        //ab dekho max of abhi nd recursive call ka result
        int new_p_part1=solve(price,width,height-1);
        int new_p_part2=solve(price,width,1);
        int result = max(abhi,new_p_part1+new_p_part2);
        p=max(p,result);
        height--;
    }
    return p;
}

int main(){
    vector<vector<int>> price={
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    int m=3;
    int n=3;
    int result=solve(price,m,n);
    cout<<result;

return 0;
}  */
/*

#include <iostream>
#include <vector>
#include <climits>

using namespace std;

int maximizeProfit(vector<vector<int>>& spotPrices, int m, int n) {
    // Create a 2D array to store the maximum profit for each subproblem
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    // Iterate over all possible subproblems
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            // Initialize the maximum profit for the current subproblem
            dp[i][j] = INT_MIN;

            // Try making horizontal cuts
            for (int k = 1; k <= i; ++k) {
                dp[i][j] = max(dp[i][j], spotPrices[k - 1][j - 1] + dp[i - k][j]);
            }

            // Try making vertical cuts
            for (int k = 1; k <= j; ++k) {
                dp[i][j] = max(dp[i][j], spotPrices[i - 1][k - 1] + dp[i][j - k]);
            }
        }
    }

    // The bottom-right corner of the dp array contains the maximum profit
    return dp[m][n];
}

int main() {
    // Example usage
    int m = 4; // Width of the marble slab
    int n = 2; // Height of the marble slab

    // Example spot prices
    vector<vector<int>> spotPrices = {
        {1, 3},
        {4, 2},
        {5, 6},
        {7, 8}
    };

    int result = maximizeProfit(spotPrices, m, n);

    cout << "Maximum Profit: " << result << endl;

    return 0;
}
 */

/*
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int maximizeProfit(vector<vector<int>>& P, int m, int n) {
    vector<vector<int>> DP(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            DP[i][j] = P[i - 1][j - 1];
            for (int horizontal_cut = 1; horizontal_cut < i; ++horizontal_cut) {
                DP[i][j] = max(DP[i][j], DP[horizontal_cut][j] + DP[i - horizontal_cut][j]);
            }
            for (int vertical_cut = 1; vertical_cut < j; ++vertical_cut) {
                DP[i][j] = max(DP[i][j], DP[i][vertical_cut] + DP[i][j - vertical_cut]);
            }
        }
    }

    return DP[m][n];
}

int main() {
    vector<vector<int>> spotPrices = {
        {1, 3},
        {4, 2},
        {5, 6},
        {7, 8}
    };
    int m = 4;
    int n = 2;

    int maxProfit = maximizeProfit(spotPrices, m, n);
    cout << "Maximum profit: " << maxProfit << endl;

    return 0;
} */

/*
#include<bits/stdc++.h>
using namespace std;
int main(){
    vector<int> hash(100,0);
    vector<int> input={1,2,3,1,5};
    for(int i=0;i<input.size();i++){
        if(hash[input[i]]!=0){
            cout<< input[i];
            break;
        }
        hash[input[i]]++;
    }
return 0;
}
 */

/*
#include <bits/stdc++.h>
using namespace std;

int solve(vector<int> a,int rcount,int dcount,int score,int pointer){
    if(pointer>=a.size()){
        return score;
    }
    //try to ding
    int ding=0;
    int ring=0;
    if(dcount<3){
        int sc=0;
        if(a[pointer]<0){
            sc=abs(a[pointer]);
        }
        else{
            sc=a[pointer]*-1;
        }
        ding=solve(a,0,dcount+1,score+sc,pointer+1);
    }
    else{
        //ring always
        ring=solve(a,rcount+1,0,score+a[pointer],pointer+1);
    }

    //try for ring
    if(rcount<3){
        int sc=0;
        ring=solve(a,rcount+1,0,score+a[pointer],pointer+1);
    }
    else{
        int sc=0;
        if(a[pointer]<0){
            sc=abs(a[pointer]);
        }
        else{
            sc=a[pointer]*-1;
        }
        ding=solve(a,0,dcount+1,score+sc,pointer+1);
    }
    score=max(ding,ring);
    return score;
}

int main() {
    // Example usage
    vector<int> A = {-10 ,-20, -30, -40, -50, -60, -70 ,-80, -90, -100, -110, -120 ,-130, -140, -150};
    int result = solve(A,0,0,0,0);

    cout << "The largest number of chickens earned by Mr. Fox is: " << result << endl;

    return 0;
}
 */
/*
#include <bits/stdc++.h>
using namespace std;
#define b bool

int solve_ring_ding(vector<int> a, int rcount, int dcount, int score, int pointer)
{
    if (pointer >= a.size())
    {
        return score;
    }
    // try to ding
    int ding = 0;
    int ring = 0;
    if (dcount < 3)
    {
        int sc = 0;
        if (a[pointer] < 0)
        {
            sc = abs(a[pointer]);
        }
        else
        {
            sc = a[pointer] * -1;
        }
        ding = solve_ring_ding(a, 0, dcount + 1, score + sc, pointer + 1);
    }
    else
    {
        // ring always
        ring = solve_ring_ding(a, rcount + 1, 0, score + a[pointer], pointer + 1);
    }

    // try for ring
    if (rcount < 3)
    {
        int sc = 0;
        ring = solve_ring_ding(a, rcount + 1, 0, score + a[pointer], pointer + 1);
    }
    else
    {
        int sc = 0;
        if (a[pointer] < 0)
        {
            sc = abs(a[pointer]);
        }
        else
        {
            sc = a[pointer] * -1;
        }
        ding = solve_ring_ding(a, 0, dcount + 1, score + sc, pointer + 1);
    }
    score = max(ding, ring);
    return score;
}

void placement_de_do(vector<pair<long int, long int>> &array, int remove_counter, vector<int> &answers, int index, int energy, int capacity_kitni_h, int juice, int &en)
{
    if (remove_counter == 0)
    {
        if (energy < en)
        {
            answers.push_back(energy);
            en = energy;
        }
        return;
    }
    if (index >= array.size() - 1)
    {
        return;
    }
    // remove krne ka try to kr yr
    int tank_capacity = array[index].first;
    int energy_kitni_lgi = array[index].second;
    b f = false;
    if (capacity_kitni_h - tank_capacity >= juice)
    {
        // kr de remove
        f = true;
        // energy+=energy_kitni_lgi;
        placement_de_do(array, remove_counter - 1, answers, index + 1, energy + energy_kitni_lgi, capacity_kitni_h - tank_capacity, juice, en);
    }

    // mt kr remove na

    if (f == false)
    {
        placement_de_do(array, remove_counter, answers, index + 1, energy, capacity_kitni_h, juice, en);
    }
    else
    {
        // energy-=energy_kitni_lgi;
        placement_de_do(array, remove_counter, answers, index + 1, energy, capacity_kitni_h, juice, en);
    }
} */
/*
int main(){
    long int n;
    cin>>n;
    vector<long int> filled;
    vector<long int> capacity;
    vector<pair<long int,long int>> array;
    long int sum=0;
    long int cap_sum=0;
    for(long int i=0;i<n;i++){
        long int ele;
        cin>>ele;
        sum+=ele;
        filled.push_back(ele);
    }
    for(long int i=0;i<n;i++){
        long int ele;
        cin>>ele;
        cap_sum=cap_sum+ele;
        capacity.push_back(ele);
    }

    for(long int i=0;i<n;i++){
        pair<long int,long int> temp;
        temp.second=filled[i];
        temp.first=capacity[i];
        array.push_back(temp);
    }
    sort(array.begin(),array.end());

    long int max_cont=0;
    long int newsum=sum;
    for(long int i=array.size()-1;i>=0;i--){
        if(sum<=0){
            break;
        }
        else{
            long int can_full=array[i].first;
            sum=sum-can_full;
            max_cont++;
        }
    }
    cout<<max_cont<<" ";
    long int energy=0;
    long int temp=max_cont;
    /*
    vector<pair<float,int>> ration_and_index;
    for(int i=0;i<n;i++){
        float r=float(array[i].second)/float(array[i].first);
        pair<float,int> temp;
        temp.first=r;
        temp.second=i;
        ration_and_index.push_back(temp);
    }
    sort(ration_and_index.begin(),ration_and_index.end());
    for(int i=0;i<ration_and_index.size();i++){
        pair<float,int> t=ration_and_index[i];
        cout<<t.first<<" ";
    }

    int enery_used=0;
    for(int i=0;i<n;i++){
        if(temp<=0){
            break;
        }
        //try to remove this
        int ind=ration_and_index[i].second;
        int capacity_kitni_h=array[ind].first;
        if(cap_sum-capacity_kitni_h >=sum){
            temp--;
            enery_used+=array[ind].second;
            cap_sum-=capacity_kitni_h;
        }
    }
    cout<<enery_used;*/

/*
for(long int i=array.size()-1;i>=0;i--){
    if(temp<=0){
        f=true;
        break;
    }
    long int filled_tank=array[i].second;
    newsum=newsum-filled_tank;
    temp--;
}

return 0;
} */

/*
#include<bits/stdc++.h>
using namespace std;
int main(){
    int test;
    cin>>test;
    for(int i=0;i<test;i++){
        int vertex;
        int edge;
        cin>>vertex;
        cin>>edge;
        unordered_map<int,vector<int>> adjlist;
        for(int j=0;j<edge;j++){
            int v1;
            int v2;
            cin>>v1;
            cin>>v2;
            adjlist[v1].push_back(v2);
        }
        for(auto &it:adjlist){
            int node=it.first;
            vector<int> conn=it.second;
            cout<<"node: "<<node<<" ";
            for(int i=0;i<conn.size();i++){
                cout<<conn[i];
            }
            cout<<endl;
        }
    }
return 0;
}
 */

/*
int main(){
   long int n;
   cin>>n;
   vector<long int> filled;
   vector<long int> capacity;
   vector<pair<long int,long int>> array;
   long int sum=0;
   long int cap_sum=0;
   for(long int i=0;i<n;i++){
       long int ele;
       cin>>ele;
       sum+=ele;
       filled.push_back(ele);
   }
   for(long int i=0;i<n;i++){
       long int ele;
       cin>>ele;
       cap_sum=cap_sum+ele;
       capacity.push_back(ele);
   }

   for(long int i=0;i<n;i++){
       pair<long int,long int> temp;
       temp.second=filled[i];
       temp.first=capacity[i];
       array.push_back(temp);
   }
   sort(array.begin(),array.end());

   long int max_cont=0;
   long int newsum=sum;
   for(long int i=array.size()-1;i>=0;i--){
       if(sum<=0){
           break;
       }
       else{
           long int can_full=array[i].first;
           sum=sum-can_full;
           max_cont++;
       }
   }
   cout<<max_cont<<" ";
   long int energy=0;
   long int temp=max_cont;
   b f=false;
   vector<int> iska_answer;
   placement_de_do(array,array.size()-max_cont,iska_answer,0,0,cap_sum,newsum);
   sort(iska_answer.begin(),iska_answer.end());
   cout<<iska_answer[0];
 */
/*
for(long int i=array.size()-1;i>=0;i--){
    if(temp<=0){
        f=true;
        break;
    }
    long int filled_tank=array[i].second;
    newsum=newsum-filled_tank;
    temp--;
}

return 0;
}  */

/*

void placement_de_do(vector<pair<long int,long int>> &array,int remove_counter,vector<int>& answers,int index,int energy,int capacity_kitni_h,int juice,int& en){
   if(remove_counter==0){
       if(energy<en){
       answers.push_back(energy);
       en=energy;
       }
       return;
   }
   if(index>=array.size()-1 ){
       return;
   }
   //remove krne ka try to kr yr
   int tank_capacity=array[index].first;
   int energy_kitni_lgi=array[index].second;
   b f=false;
   if(capacity_kitni_h-tank_capacity >=juice){
       // kr de remove
       f=true;
       //energy+=energy_kitni_lgi;
       placement_de_do(array,remove_counter-1,answers,index+1,energy+energy_kitni_lgi,capacity_kitni_h-tank_capacity,juice,en);
   }


   // mt kr remove na

   if(f==false){
       placement_de_do(array,remove_counter,answers,index+1,energy,capacity_kitni_h,juice,en);
   }
   else{
      //energy-=energy_kitni_lgi;
       placement_de_do(array,remove_counter,answers,index+1,energy,capacity_kitni_h,juice,en);
   }

}

int main(){
   long int n;
   cin>>n;
   vector<long int> filled;
   vector<long int> capacity;
   vector<pair<long int,long int>> array;
   long int sum=0;
   long int cap_sum=0;
   for(long int i=0;i<n;i++){
       long int ele;
       cin>>ele;
       sum+=ele;
       filled.push_back(ele);
   }
   for(long int i=0;i<n;i++){
       long int ele;
       cin>>ele;
       cap_sum=cap_sum+ele;
       capacity.push_back(ele);
   }

   for(long int i=0;i<n;i++){
       pair<long int,long int> temp;
       temp.second=filled[i];
       temp.first=capacity[i];
       array.push_back(temp);
   }
   sort(array.begin(),array.end());

   long int max_cont=0;
   long int newsum=sum;
   for(long int i=array.size()-1;i>=0;i--){
       if(sum<=0){
           break;
       }
       else{
           long int can_full=array[i].first;
           sum=sum-can_full;
           max_cont++;
       }
   }
   cout<<max_cont<<" ";
   long int energy=0;
   long int temp=max_cont;
   b f=false;
   vector<int> iska_answer;
   int en=INT32_MAX;
   placement_de_do(array,array.size()-max_cont,iska_answer,0,0,cap_sum,newsum,en);
   //sort(iska_answer.begin(),iska_answer.end());
   cout<<iska_answer[iska_answer.size()-1];

   /*
   for(long int i=array.size()-1;i>=0;i--){
       if(temp<=0){
           f=true;
           break;
       }
       long int filled_tank=array[i].second;
       newsum=newsum-filled_tank;
       temp--;
   }

return 0;
} */

/*
#include<bits/stdc++.h>
using namespace std;

void placement_de_do(const vector<pair<long int, long int>>& array, int remove_counter, vector<int>& answers, int index, int energy, int capacity_kitni_h, int juice, int& en, vector<vector<vector<int>>>& memo) {
    if (remove_counter == 0) {
        if (energy < en) {
            answers.push_back(energy);
            en = energy;
        }
        return;
    }
    if (index >= array.size() - 1) {
        return;
    }
    if (memo[remove_counter][index][energy] != -1) {
        return;
    }

    int tank_capacity = array[index].first;
    int energy_kitni_lgi = array[index].second;

    // Try removing the tank
    if (capacity_kitni_h - tank_capacity >= juice) {
        placement_de_do(array, remove_counter - 1, answers, index + 1, energy + energy_kitni_lgi, capacity_kitni_h - tank_capacity, juice, en, memo);
    }

    // Don't remove the tank
    placement_de_do(array, remove_counter, answers, index + 1, energy, capacity_kitni_h, juice, en, memo);

    memo[remove_counter][index][energy] = 1;
}

int main() {
    long int n;
    cin >> n;
    vector<long int> filled;
    vector<long int> capacity;
    vector<pair<long int, long int>> array;
    long int sum = 0;
    long int cap_sum = 0;
    for (long int i = 0; i < n; i++) {
        long int ele;
        cin >> ele;
        sum += ele;
        filled.push_back(ele);
    }
    for (long int i = 0; i < n; i++) {
        long int ele;
        cin >> ele;
        cap_sum = cap_sum + ele;
        capacity.push_back(ele);
    }

    for (long int i = 0; i < n; i++) {
        pair<long int, long int> temp;
        temp.second = filled[i];
        temp.first = capacity[i];
        array.push_back(temp);
    }
    sort(array.begin(), array.end());

    long int max_cont = 0;
    long int newsum = sum;
    for (long int i = array.size() - 1; i >= 0; i--) {
        if (sum <= 0) {
            break;
        }
        else {
            long int can_full = array[i].first;
            sum = sum - can_full;
            max_cont++;
        }
    }
    cout << max_cont << " ";
    long int energy = 0;
    long int temp = max_cont;
    vector<int> iska_answer;
    int en = INT_MAX;
    vector<vector<vector<int>>> memo(max_cont + 1, vector<vector<int>>(n, vector<int>(cap_sum + 1, -1)));
    placement_de_do(array, array.size() - max_cont, iska_answer, 0, 0, cap_sum, newsum, en, memo);

    cout << iska_answer[iska_answer.size() - 1];

    return 0;
}
 */
/*
#include <bits/stdc++.h>
using namespace std;
#define b bool

void placement_de_do(vector<pair<long int,long int>> &array,int remove_counter,vector<int>& answers,int index,int energy,int capacity_kitni_h,int juice,int& en){
   if(remove_counter==0){
       if(energy<en){
       answers.push_back(energy);
       en=energy;
       }
       return;
   }
   if(index>=array.size()-1 ){
       return;
   }
   //remove krne ka try to kr yr
   int tank_capacity=array[index].first;
   int energy_kitni_lgi=array[index].second;
   b f=false;
   if(capacity_kitni_h-tank_capacity >=juice){
       // kr de remove
       f=true;
       //energy+=energy_kitni_lgi;
       placement_de_do(array,remove_counter-1,answers,index+1,energy+energy_kitni_lgi,capacity_kitni_h-tank_capacity,juice,en);
   }


   // mt kr remove na

   if(f==false){
       placement_de_do(array,remove_counter,answers,index+1,energy,capacity_kitni_h,juice,en);
   }
   else{
      //energy-=energy_kitni_lgi;
       placement_de_do(array,remove_counter,answers,index+1,energy,capacity_kitni_h,juice,en);
   }

}

void fun1(vector<int> &arr){
    for(int i=0;i<1024;i++){
        arr.push_back(0);
    }
}

stack<long long int> solve_kr(vector<int>& na,vector<int>& dp,int upper,int lower){
    stack<long long int> answer;
    long int old=dp[lower];
    long long int ind=lower|upper;
    long long int sum=na[ind]+old;
    answer.push(sum);
    return answer;
}


int calculate(vector<int>& arr, int x) {
    vector<int> dp;
    fun1(dp);
    /*
    for(int i=0;i<1024;i++){
        cout<<dp[i];
    }

    dp[0] = 1;

    for(int i=0;i<arr.size();i++){
        vector<int> new_array=dp;
        long int j=0;
        while(j<1024){
            int arr_val=arr[i];
            stack<long long int> result=solve_kr(new_array,dp,arr_val,j);
            int ind=j|arr_val;
            long long int st_val=result.top();
            new_array[ind]=st_val %1000000007;
            j++;
        }
        dp=new_array;
    }
    return dp[x];

}

int main()
{
    int q;
    cin >> q;
    vector<int> num;
    for (int i = 0; i < q; i++)
    {
        int mode;
        int val;
        cin >> mode;
        cin >> val;
        if (mode == 1)
        {
            num.push_back(val);
        }
        else if(mode==2){
            long int result=calculate(num,val);
            cout<<result<<endl;
        }
        else
        {   int temp=mode;
            long temp2=mode+val;
            cout<<temp;
            continue;
        }
    }
    return 0;
} */
/*
#include<bits/stdc++.h>
using namespace std;
int main(){
    int c;
    cin>>c;
    vector<int> capacity;
    vector<int> filled;
    for(int i=0;i<c;i++){
        int ele;
        cin>>ele;
        filled.push_back(ele);
    }
    for(int i=0;i<c;i++){
        int ele;
        cin>>ele;
        capacity.push_back(ele);
    }


return 0;
} */

/*
#include<bits/stdc++.h>
using namespace std;

void fun1(vector<int> &arr){
    for(int i=0;i<1024;i++){
        arr.push_back(0);
    }
}

stack<long long int> solve_kr(vector<int>& na,vector<int>& dp,int upper,int lower){
    stack<long long int> answer;
    long int old=dp[lower];
    long long int ind=lower|upper;
    long long int sum=na[ind]+old;
    answer.push(sum);
    return answer;
}


int calculate(vector<int>& arr, int x) {
    vector<int> dp;
    fun1(dp);
    /*
    for(int i=0;i<1024;i++){
        cout<<dp[i];
    }*/

/*
int arr_val=arr[i];
stack<long long int> result=solve_kr(new_array,dp,arr_val,j);
int ind=j|arr_val;
long long int st_val=result.top();
new_array[ind]=st_val %1000000007;
dp[j | i] = (dp[j | i] + 1LL * dp[j] * arr[i]) %1000000007 ;
j++;
}
//dp=new_array;
}
return dp[x];

}

void solve(vector<int> &num,vector<int> &dp){
for(int i=0;i<1024;i++){

}

}
int main()
{
int q;
cin >> q;
vector<int> num;
vector<int> counts(1024,0);
for (int i = 0; i < q; i++)
{
int mode;
int val;
cin >> mode;
cin >> val;
if (mode == 1)
{
num.push_back(val);
counts[val]++;

}
else if(mode==2){
long int result=calculate(counts,val);
cout<<result<<endl;
}
else
{   int temp=mode;
long temp2=mode+val;
cout<<temp;
continue;
}
}
return 0;
} */
/* 
#include <bits/stdc++.h>
using namespace std;
#define b bool

int solve_ring_ding(vector<int> a, int rcount, int dcount, int score, int pointer)
{
    if (pointer >= a.size())
    {
        return score;
    }
    // try to ding
    int ding = 0;
    int ring = 0;
    if (dcount < 3)
    {
        int sc = 0;
        if (a[pointer] < 0)
        {
            sc = abs(a[pointer]);
        }
        else
        {
            sc = a[pointer] * -1;
        }
        ding = solve_ring_ding(a, 0, dcount + 1, score + sc, pointer + 1);
    }
    else
    {
        // ring always
        ring = solve_ring_ding(a, rcount + 1, 0, score + a[pointer], pointer + 1);
    }

    // try for ring
    if (rcount < 3)
    {
        int sc = 0;
        ring = solve_ring_ding(a, rcount + 1, 0, score + a[pointer], pointer + 1);
    }
    else
    {
        int sc = 0;
        if (a[pointer] < 0)
        {
            sc = abs(a[pointer]);
        }
        else
        {
            sc = a[pointer] * -1;
        }
        ding = solve_ring_ding(a, 0, dcount + 1, score + sc, pointer + 1);
    }
    score = max(ding, ring);
    return score;
}

void placement_de_do(vector<pair<long int, long int>> &array, int remove_counter, vector<int> &answers, int index, int energy, int capacity_kitni_h, int juice)
{
    if (remove_counter == 0)
    {
        answers.push_back(energy);
        return;
    }
    if (index >= array.size() - 1)
    {
        return;
    }
    // remove krne ka try to kr yr
    int tank_capacity = array[index].first;
    int energy_kitni_lgi = array[index].second;
    b f = false;
    if (capacity_kitni_h - tank_capacity >= juice)
    {
        // kr de remove
        f = true;
        // energy+=energy_kitni_lgi;
        placement_de_do(array, remove_counter - 1, answers, index + 1, energy + energy_kitni_lgi, capacity_kitni_h - tank_capacity, juice);
    }

    // mt kr remove na
    if (f == false)
    {
        placement_de_do(array, remove_counter, answers, index + 1, energy, capacity_kitni_h, juice);
    }
    else
    {
        // energy-=energy_kitni_lgi;
        placement_de_do(array, remove_counter, answers, index + 1, energy, capacity_kitni_h, juice);
    }
}
unordered_map<string, pair<int, int>> memo;
pair<int, int> solve(vector<pair<long int, long int>> &array, int index, int num_of_cont, int juice, int min_cap, int max_juice, int max_cont)
{
    string temp = to_string(index) + to_string(num_of_cont) + to_string(juice);
    auto it = memo.find(temp);
    if (it != memo.end())
    {
        return memo[temp];
    }
    if (index >= array.size())
    {
        if (num_of_cont == max_cont)
        {   memo[temp]={num_of_cont,juice};
            return {num_of_cont, juice};
        }
        memo[temp]={INT32_MAX,INT32_MAX};
        return {INT32_MAX, INT32_MAX};
    }

    // pick kr de
    int juice_kitna_h = array[index].second;
    pair<int, int> pick_kiya_h_isko = solve(array, index + 1, num_of_cont + 1, juice - juice_kitna_h, min_cap, max_juice, max_cont);

    // pick hi mt kr
    pair<int, int> pick_nhi_kra_h_isko = {INT32_MAX, INT32_MAX};
    if (min_cap - array[index].first >= max_juice)
    {
        pick_nhi_kra_h_isko = solve(array, index + 1, num_of_cont, juice, min_cap - array[index].first, max_juice, max_cont);
    }

    if (pick_kiya_h_isko.first < pick_nhi_kra_h_isko.first)
    {   memo[temp]=pick_kiya_h_isko;
        return pick_kiya_h_isko;
    }
    else if (pick_kiya_h_isko > pick_nhi_kra_h_isko)
    {   memo[temp]=pick_nhi_kra_h_isko;
        return pick_nhi_kra_h_isko;
    }
    else
    {
        if (pick_kiya_h_isko.second < pick_nhi_kra_h_isko.second)
        {   memo[temp]=pick_kiya_h_isko;
            return pick_kiya_h_isko;
        }
        else
        {   memo[temp]=pick_nhi_kra_h_isko;
            return pick_nhi_kra_h_isko;
        }
    }
}

int main()
{
    long int n;
    cin >> n;
    vector<long int> filled;
    vector<long int> capacity;
    vector<pair<long int, long int>> array;
    long int sum = 0;
    long int cap_sum = 0;
    for (long int i = 0; i < n; i++)
    {
        long int ele;
        cin >> ele;
        sum += ele;
        filled.push_back(ele);
    }
    for (long int i = 0; i < n; i++)
    {
        long int ele;
        cin >> ele;
        cap_sum = cap_sum + ele;
        capacity.push_back(ele);
    }

    for (long int i = 0; i < n; i++)
    {
        pair<long int, long int> temp;
        temp.second = filled[i];
        temp.first = capacity[i];
        array.push_back(temp);
    }
    int index = 0;
    int num_of_containers = 0;
    int juice = sum;
    int min_cap = sum;
    long int max_cont = 0;
    long int newsum = sum;
    for (long int i = array.size() - 1; i >= 0; i--)
    {
        if (sum <= 0)
        {
            break;
        }
        else
        {
            long int can_full = array[i].first;
            sum = sum - can_full;
            max_cont++;
        }
    }
    pair<int, int> answer = solve(array, index, num_of_containers, juice, cap_sum, sum, max_cont);
    cout << answer.first <<" "<< answer.second;

    return 0;
}
 */

#include<bits/stdc++.h>
using namespace std;

int count_kr_de(unordered_map<int, vector<int>> &graph,int destination){
    
}


int main() {
    int testcase;
    cin >> testcase;
    int n, m;
 
    for (int t = 0; t < testcase; ++t) {
        cin >> n >> m;
 
        unordered_map<int, vector<int>> graph;
        if(m==0){
            for(int i = 1; i<=n; i++){
                if(i==1){
                    cout<<1<<" ";
                }
                else{
                    cout<<0<<" ";
                }
            }
        }
        // Input the connections between vertices
        for (int i = 0; i < m; ++i) {
            int u, v;
            cin >> u >> v;
 
            graph[u].push_back(v);
        }
        for(int i=1;i<=n;i++){
            int count=count_kr_de(graph,i);
        }
    }
 
    return 0;
}